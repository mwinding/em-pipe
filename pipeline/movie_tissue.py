"""Promotional movie of a whole volume, showing only the data: the stack of slices growing from the last
slice to the first (each new slice on the face towards the camera: the reverse of the order they were
imaged, so the block visibly materialises), the complete block with the camera
circling it, then the resin and empty background turning transparent until only the tissue is left,
still circling. With ``fly_through``, the movie then ends on the fly-through (pipeline.movie, the
``movie:`` settings), tissue only: the volume is cut away in front of the fly-through's first slice,
from the first slice imaged onwards (the order the slices were imaged and milled away), as the camera
comes face-on to it; then the fly-through's zoom and scroll, still in 3D and tissue only: the camera
face-on to the scrolling cut, the slice on it from the level the zoom needs, its tissue only, and through
the resin the tissue behind it.

Everything up to the fly-through is drawn from one coarse pyramid level held in memory (``level``:
s5, 256 nm voxels for 8 nm data, ~0.4 GB for 35h) by a 2D-texture volume renderer: the level's
slices along the axis closest to the view direction are warped onto the frame in perspective and
composited back to front (two stacks blended where two axes are about as close). A voxel shows its
grey value, unchanged; only its opacity varies: tissue is opaque, the background (data outside the
tissue mask: resin) is opaque in the block and turns transparent, and voxels without data (0, outside
the imaged area) are always transparent, so the block shows its true, stepped outline. The voxels on
the imaged area's edge, only partly imaged and so darkened by the pyramid's averaging with the zeros
outside it, are left out (``trim_edge_px``). No shading, lighting or colour.

The tissue mask is 3D, made on the same level (``tissue_mask``): resin is where the image is flat
along y or along x (tissue has fine texture in every direction; curtaining stripes, slice-to-slice
brightness changes and the edges of the imaged area do not) over large regions, or over small ones
that are as bright as resin; and, at the tissue's edge, where a voxel is closer to the local resin
grey than to the local tissue grey. Specks (< 1 um^3) change sides; no holes are filled and nothing
is smoothed. It is cached in the movie's work folder, with a QC image in qc/.

    python -m pipeline.movie_tissue --config C mask      # the tissue mask and qc/<name>_mask.png
    python -m pipeline.movie_tissue --config C frames    # PNG frames -> work/movie/<name>/
    python -m pipeline.movie_tissue --config C encode    # ffmpeg -> output_dir/<name>.mp4

Coordinates are continuous s0 voxels (x, y, z) as in pipeline.movie: the camera looks along +z at
yaw 0 (the first slice facing it) and the orbit turns it about the y axis. z is the order of imaging
(pipeline.check numbers slices by their time stamps).
"""

import json
import logging
import math
import os
import shlex
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage

from . import cli, movie, serve
from .config import deep_merge, qc_path, step_dir
from .movie import camera, open_levels, project

log = logging.getLogger(__name__)

DEFAULTS = {
    "tissue_movie": {
        "volume": None,            # OME-Zarr in output_dir to film; null: <name>.ome.zarr
        "name": None,              # -> output_dir/<name>.mp4; null: <volume stem>_tissue_movie
        "size": [1920, 1080],      # frame width, height (px)
        "view_rotation_deg": 0,    # rotates the picture in the frame plane: 0 or 180 (as movie.view_rotation_deg)
        "fps": 30,
        "level": 5,                # pyramid level rendered and masked (5: 32x, 256 nm voxels at 8 nm)
        "mask_volume": None,       # the tissue mask (and what is imaged) from this volume in output_dir (the same
                                   # geometry), e.g. the plain one for a CLAHE version; null: the volume itself
        # Timeline (s). The orbit runs through it, from rest and back to rest.
        "build_s": 7.0,            # the stack growing slice by slice, from the last slice to the first, each
                                   # new slice on the face towards the camera (0: none)
        "first_slice_s": 0.6,      # the first slice appears over this long (then the rest)
        "build_order": "reverse",  # reverse: from the last slice to the first (z = 0), each new slice on the
                                   # z = 0 face; imaging: as imaged, from z = 0 on, each new slice on the far
                                   # face (for a camera on that side: yaw beyond +-90)
        "block_s": 0.0,            # the whole block, before the next phase (0: it starts as the build ends)
        "fade_s": 5.0,             # the background turning transparent
        "tissue_s": 8.0,           # the tissue alone
        "fade_out_s": 0.0,         # to black at the very end (part of tissue_s; 0: none; not with fly_through)
        # Camera: orbits the volume's centre, looking at it (as pipeline.movie.camera: yaw 0 looks along +z at
        # the first slice, negative yaw from -x, on screen right at view_rotation_deg 180)
        "yaw_deg": [-60, 0],       # the camera's turn about the y axis at the start (the build, seen from three
                                   # quarters) and after the turn to the front (yaw 0: facing z = 0)
        "tilt_deg": [22, 22],      # ... and its height there (degrees, towards up on screen)
        "drift_deg": [3, 10],      # yaw drift during the build and the block (before the turn), and after it
                                   # to the end (0 with fly_through: the approach follows)
        "turn_s": 6.0,             # the turn to the front: speeding up from rest, easing in
        "fade_first": False,       # fade the resin before the turn (else after it)
        "blend_s": 3.0,            # each phase starts this long before the one before it ends: the build -> the
                                   # turn -> the cut's sweep (and with fade_first the fade -> the turn), so one
                                   # slows down as the next speeds up (their eases are about this long)
        "fade_start_s": None,      # the fade starts this long after the slicing stops (the build's last frame
                                   # advancing by a voxel of `level` or more; null: blend_s before the turn
                                   # ends; with fade_first, the turn starts blend_s before the fade ends) ...
        "fade_at_yaw_deg": None,   # ... or as the turning camera passes this yaw
        "ease_s": 3.0,             # the drift after the turn eases out over this long at the end; the growth
                                   # speeds up from rest and slows to rest over this long (at most half the
                                   # build)
        "fov_deg": 40,             # field of view across the frame's long side (the width, landscape; in portrait
                                   # the same lens turned)
        "fill": 0.84,              # the block's share of the frame height (or width): sets the distance
        "push_in": 0.1,            # the camera comes this fraction closer from the fade's start to the orbit's end
        # Opacity
        "resin_depth": 100,        # the background clears smoothly: how deep one sees into it grows from 0
                                   # (opaque) to unlimited, this many voxels (of `level`; 100: 25 um at s5)
                                   # halfway; tissue just below the surface shows first
        "trim_edge_px": 1,         # outer voxel layers of the imaged area left out: only partly imaged, the
                                   # pyramid darkened them (averaged with the no-data zeros around)
        "crop": {},                # a display crop (s0 voxels): what lies below x0, y0, z0 or above x1, y1, z1
                                   # is not shown, e.g. {x0: 384}
        "blend_axes": 0.06,        # two slice stacks are blended while their axes' alignment with the view
                                   # differs by less than this (cosines): no jump when the stack switches
        # Ending on the fly-through (the `movie:` settings, pipeline.movie: target, zoom, scroll; rotate_deg 0
        # and zoom_out_s 0; size, fps and view_rotation_deg as here)
        "fly_through": False,
        "approach_s": 5.0,         # then the camera comes face-on to the cut while it sweeps from the first
                                   # slice into the fly-through's scroll (no stop: its speed at the end is the
                                   # scroll's)
        "zoom_overlap_s": 1.0,     # the fly-through's zoom-in starts this long before the approach ends (the
                                   # approach blending into its moving camera: no pause, no pull-back)
        "handover_s": 1.0,         # the slice on the cut from the finer level fades in over the first this
                                   # long of the fly-through
        "zoom_out_s": 0.0,         # then the fly-through zooms back out over this long (0: it ends with the
                                   # zoom), the scroll reversing: the cut runs back to z = 0, the removed
                                   # slices reappearing on the face towards the camera ...
        "zoom_out_cut": 0.9,       # ... this far into it, the outro takes over (the pull-back, the spin):
        "back_rate": 60,           # the scroll-back: slices/s per zoom (voxels per frame px, on a frame ref_width
        "ref_width": 1920,         # px along its long side: the same speeds for any frame size)
        "back_boost_zoom": 10,     # ... and faster when far out: x (1 + (zoom / this)^2) (null: no boost)
        "back_turn_s": 0.5,        # ... reversing from the forward scroll within this long
        "back_land_s": 0.4,        # ... slowing to rest over about this long as the last slice lands
        "spin_s": 12.0,            # the camera, face-on to the whole tissue volume, turning once round it,
                                   # speeding up from rest and easing to rest ...
        "pull_min_s": 3.0,         # the pull-back to the whole volume: at least this long (longer if the
                                   # scroll-back needs it: it ends as the last slice lands) ...
        "spin_overlap_s": 3.0,     # ... the spin starting this long before it ends (pull_min_s: with it),
        "spin_in_s": 3.0,          # ... speeding up over this long, easing to rest over the rest of spin_s
        "spin_deg": 360,           # ... by this much (-360: the other way round, e.g. for a volume stored
                                   # rotated 180 degrees, with view_rotation_deg 0 and the yaws negated)
        "spin_tilt_deg": 15,       # ... rising this far as it starts turning
        "end_fade_s": 1.5,         # the end: to black over this long
        "mask": {
            "max_texture": 1.5,    # rms (greys) of the fine texture, along y or along x, below which a voxel
                                   # is flat (resin)
            "smooth_px": 1.5,      # sigma (voxels) of the smoothing along y (x) whose residual is the fine texture
            "texture_sigma": 1.0,  # sigma (voxels, in the slice) of the rms
            "edge_px": 2,          # px at the edge of the imaged area (partly imaged: darker) whose texture is
                                   # taken from inside it
            "min_resin": 5000,     # flat regions this large (voxels; 84 um^3 at s5) reaching the edge of the
                                   # volume or of the imaged area are resin ...
            "pocket_split": 0.8,   # ... other flat regions only if this much of the way from the local tissue
                                   # grey to the local resin grey (large pale cells and nuclei are flat at s5
                                   # too, and pale cells nearly as bright as resin where the image is bright)
            "rim_px": 16,          # tissue within this many px (per slice) of resin is checked against the
                                   # local resin and tissue greys: the texture spreads a few px into resin
            "rim_split": 0.5,      # ... and is resin if brighter than this fraction of the way to resin
                                   # (and joined to resin through such pixels)
            "min_contrast": 20,    # ... where resin is at least this much brighter than tissue
            "min_speck": 64,       # tissue or resin pieces smaller than this (voxels; 64: 1 um^3 at s5) are
                                   # specks and change sides
            "bleed_px": 0,         # the dark bleed at the imaged area's boundary: within this many px of it
            "bleed_split": 0.5,    # ... darker than this fraction of the local tissue grey, joined to it
                                   # (0: off; on 35h at 12 px it also took darker, textured tiles at the edge)
        },
        "mask_level": None,        # (previews) the level whose tissue mask is pooled for this level's
        "preview_scale": 1,        # > 1: a preview this many times smaller (the same view, frames from coarser
                                   # levels; kept apart from the full-size frames)
        "workers": 1,              # frames rendered in parallel processes
        "crf": 18,                 # x264 quality: lower is better and bigger (18: visually lossless)
        "ffmpeg": "ffmpeg",
    },
}


def volume_path(cfg):
    m = cfg["tissue_movie"]
    return Path(cfg["output_dir"]) / (m["volume"] or f"{cfg['name']}.ome.zarr")


def mask_volume_path(cfg):
    """The volume the tissue mask is made from: ``mask_volume`` (the same geometry), else the volume itself."""
    m = cfg["tissue_movie"]
    return Path(cfg["output_dir"]) / m["mask_volume"] if m.get("mask_volume") else volume_path(cfg)


def movie_name(cfg):
    m = cfg["tissue_movie"]
    return m["name"] or volume_path(cfg).name.removesuffix(".ome.zarr") + "_tissue_movie"


def work_dir(cfg):
    return step_dir(cfg, "movie", movie_name(cfg))


# ---------------------------------------------------------------------------------------------------
# The 3D tissue mask


def fill_invalid(img, valid):
    """``img`` with its pixels outside ``valid`` set to the value of the nearest valid pixel."""
    if valid.all() or not valid.any():
        return img
    iy, ix = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return img[iy, ix]


def _components(mask, keep):
    """``mask`` with only its 3D connected components (face-connected) for which ``keep(sizes)`` is true."""
    lab, n = ndimage.label(mask)
    if not n:
        return mask
    sizes = np.bincount(lab.ravel())
    ok = keep(sizes)
    ok[0] = False
    return ok[lab]


def _levels(img, valid, resin, flat, disk, fallback):
    """(grey, local resin grey, local tissue grey) of one slice: ``img`` lightly smoothed, and the local
    means of the ``resin`` pixels and of the tissue at least ``disk`` from anything flat (``fallback``:
    the resin grey where there is no resin in the slice)."""
    g = cv2.GaussianBlur(img, (0, 0), 0.7)
    core = cv2.erode((valid & ~flat).astype(np.uint8), disk) > 0
    if not core.any():
        core = valid & ~flat

    def local(m, sigma, default):   # local mean of g over m; far from m, the mean over m
        if not m.any():
            return np.full(g.shape, default, np.float32)
        num = cv2.GaussianBlur(g * m, (0, 0), sigma)
        den = cv2.GaussianBlur(m.astype(np.float32), (0, 0), sigma)
        return (num + 1e-3 * float(np.median(g[m]))) / (den + 1e-3)

    resin_grey = local(resin, 20.0, fallback)
    return g, resin_grey, local(core, 10.0, float(np.median(g[valid])) if valid.any() else 0.0)


def _flat(img, weight, smooth_px, sigma, max_texture):
    """Where a slice is flat along y or along x: the image minus its local mean along that axis (over
    ``weight`` pixels only) has a local rms (over them, ``sigma``) below ``max_texture``."""
    w = weight.astype(np.float32)
    flat = np.zeros(img.shape, bool)
    for axis in (0, 1):
        num = ndimage.gaussian_filter1d(img * w, smooth_px, axis=axis)
        den = ndimage.gaussian_filter1d(w, smooth_px, axis=axis)
        fine = (img - num / np.maximum(den, 1e-3)) * w
        rms2 = cv2.GaussianBlur(fine * fine, (0, 0), sigma) / np.maximum(cv2.GaussianBlur(w, (0, 0), sigma), 1e-3)
        flat |= rms2 < max_texture ** 2
    return flat


def edge_bleed(vol, tissue, depth_px=12, split=0.5):
    """The dark bleed at the imaged area's boundary (and the volume's own x and y faces, where the
    imaged area reaches them): per slice, tissue voxels within ``depth_px`` of the boundary that are
    darker than ``split`` of the local tissue grey (that of the tissue further in) and joined to the
    boundary through such voxels. Darkened, partly imaged or off-sample areas with streaks that pass
    as texture; nothing away from the boundary (a step from the local tissue grey, not darkness alone)."""
    out = np.zeros(vol.shape, bool)
    for z in range(len(vol)):
        g, t = vol[z].astype(np.float32), tissue[z]
        if not t.any():
            continue
        lab, n = ndimage.label(np.pad(g == 0, 1, constant_values=True))       # the outside: zeros joined
        outside = (lab == lab[0, 0])[1:-1, 1:-1] if n else np.zeros(g.shape, bool)   # to the border
        d = ndimage.distance_transform_edt(np.pad(~outside, 1, constant_values=False))[1:-1, 1:-1]
        deep = t & (d > depth_px)
        if deep.sum() < 100:
            continue
        num = cv2.GaussianBlur(g * deep, (0, 0), 15.0)
        den = cv2.GaussianBlur(deep.astype(np.float32), (0, 0), 15.0)
        local = (num + 1e-3 * float(np.median(g[deep]))) / (den + 1e-3)
        dark = t & (d <= depth_px) & (g < split * local)
        lab, n = ndimage.label(dark)
        if n:
            touching = np.unique(lab[dark & (d <= 1.5)])
            out[z] = np.isin(lab, touching[touching > 0])
    return out


def tissue_mask(vol, max_texture=1.5, smooth_px=1.5, texture_sigma=1.0, edge_px=2, min_resin=5000, rim_px=16,
                rim_split=0.5, pocket_split=0.8, min_contrast=20, min_speck=64, bleed_px=0, bleed_split=0.5):
    """3D tissue mask (uint8: 255 tissue, 0 not) of a coarse volume (z, y, x uint8; 0: no data).

    Resin is flat: the image minus its local mean along y, or along x, has a local rms (in the slice)
    below ``max_texture``. Tissue has fine texture in every direction; curtaining stripes run along y,
    and a slice's brightness and an edge of the imaged area along a slice's top or bottom are constant
    along x. As pipeline.background (which uses y only), but in 3D and without brightness; the
    ``edge_px`` px at the edge of the imaged area (partly imaged at this scale, so darker) are left out
    of the texture and take that of the image next to them. Flat regions (face-connected in 3D) of at
    least ``min_resin`` voxels that reach the volume's faces or the edge of the imaged area are resin
    (the resin around the sample); other flat regions (lumens and pockets, but also large pale cells
    and nuclei, which are flat at this scale too) only if, on average, more than ``pocket_split`` of
    the way from the local tissue grey to the local resin grey (that of the former). The texture of
    tissue reaches a few voxels into the resin next to it, so within ``rim_px`` of resin (per slice)
    pixels more than ``rim_split`` of the way to the resin grey, joined to resin through such pixels,
    are resin too. Resin and tissue pieces of fewer than ``min_speck`` voxels are specks and change
    sides. Optionally (``bleed_px``), the dark bleed at the imaged area's boundary is not tissue
    (``edge_bleed``). Nothing else: no holes are filled, nothing is smoothed or opened."""
    vol = np.asarray(vol)
    valid = vol > 0
    img = np.empty(vol.shape, np.float32)
    flat = np.zeros(vol.shape, bool)
    square = np.ones((3, 3), np.uint8)
    for z in range(len(vol)):
        # The texture from the imaged pixels only, leaving out edge_px at the edge of the imaged area
        # (partly imaged, darker); the greys for the rim with those pixels filled from inside.
        inner = cv2.erode(valid[z].astype(np.uint8), square, iterations=edge_px) > 0 if edge_px else valid[z]
        inner = inner if inner.any() else valid[z]
        flat[z] = _flat(vol[z].astype(np.float32), inner, smooth_px, texture_sigma, max_texture) & valid[z]
        img[z] = fill_invalid(vol[z].astype(np.float32), inner)
    lab, n = ndimage.label(flat)
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    sizes[0] = 0
    outer = np.zeros(n + 1, bool)                 # reaching the volume's faces or the edge of the imaged area
    for face in (lab[0], lab[-1], lab[:, 0], lab[:, -1], lab[:, :, 0], lab[:, :, -1]):
        outer[np.unique(face)] = True
    outer[np.unique(lab[ndimage.binary_dilation(~valid) & valid])] = True
    outer &= sizes >= min_resin
    outer[0] = False
    resin = outer[lab]
    pockets = np.where((sizes[lab] >= min_speck) & ~resin, lab, 0)
    del lab
    fallback = float(np.median(vol[resin])) if resin.any() else 255.0
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rim_px + 1, 2 * rim_px + 1))

    def levels(z):     # (where resin is clearly brighter than tissue, grey: 0 local tissue, 1 local resin)
        g, r, t = _levels(img[z], valid[z], resin[z], flat[z], disk, fallback)
        return r - t > min_contrast, (g - t) / np.maximum(r - t, 1e-3)

    brightness, count = np.zeros(n + 1), np.zeros(n + 1)
    for z in range(len(vol)):
        p = pockets[z]
        if p.any():
            ok, level = levels(z)
            sel = p > 0
            brightness += np.bincount(p[sel], weights=np.where(ok[sel], level[sel], 0.0), minlength=n + 1)
            count += np.bincount(p[sel], minlength=n + 1)
    bright_pockets = brightness > pocket_split * np.maximum(count, 1)
    bright_pockets[0] = False
    bg = resin | bright_pockets[pockets]
    del pockets
    for z in range(len(vol)):       # the rim: grown from all resin, levels from the large regions only
        if bg[z].any():
            ok, level = levels(z)
            band = (cv2.dilate(bg[z].astype(np.uint8), disk) > 0) & valid[z] & ~bg[z]
            joined, k = ndimage.label((band & (level > rim_split) & ok) | bg[z])
            keep = np.zeros(k + 1, bool)
            keep[np.unique(joined[bg[z]])] = True
            keep[0] = False
            bg[z] = keep[joined]
    del img
    tissue = _components(valid & ~bg, lambda sizes: sizes >= min_speck)
    tissue |= valid & ~_components(valid & ~tissue, lambda sizes: sizes >= min_speck)
    if bleed_px:
        tissue &= ~edge_bleed(vol, tissue, bleed_px, bleed_split)
    return np.where(tissue, 255, 0).astype(np.uint8)


def mask_qc(vol, mask, path, n_xy=6):
    """QC image: ``n_xy`` x-y slices, three y-z planes (columns along z) and three x-z planes (rows along z),
    background tinted red and the tissue outlined in yellow."""
    nz, ny, nx = vol.shape
    tissue = mask >= 128

    def over(img, m):
        rgb = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_GRAY2BGR)
        bg = ~m
        rgb[bg] = (0.6 * rgb[bg] + [0, 0, 100]).clip(0, 255).astype(np.uint8)
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(rgb, contours, -1, (0, 255, 255), 1)
        return rgb

    def label(img, text):
        cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        return img

    def row(tiles, height):
        tiles = [cv2.resize(t, (round(t.shape[1] * height / t.shape[0]), height), interpolation=cv2.INTER_AREA)
                 for t in tiles]
        return np.concatenate(tiles, 1)

    zs = np.linspace(0, nz - 1, n_xy + 2)[1:-1].round().astype(int)
    xy = [label(over(vol[z], tissue[z]), f"z {z}") for z in zs]
    yz = [label(over(vol[:, :, x].T, tissue[:, :, x].T), f"x {x}") for x in (nx // 4, nx // 2, 3 * nx // 4)]
    xz = [label(over(vol[:, y], tissue[:, y]), f"y {y}") for y in (ny // 4, ny // 2, 3 * ny // 4)]
    rows = [row(xy[:len(xy) // 2], 400), row(xy[len(xy) // 2:], 400), row(yz, 400), row(xz, 260)]
    width = max(r.shape[1] for r in rows)
    sheet = np.concatenate([np.pad(r, ((0, 4), (0, width - r.shape[1]), (0, 0))) for r in rows], 0)
    cli.atomic_write(path, lambda p: cv2.imwrite(p, sheet))
    return path


def load_scene(cfg, overwrite_mask=False):
    """The Scene at the movie's level: the level read once from the pyramid, the tissue mask (with its
    QC image, qc/<name>_mask.png) made once, and the level as shown (``imaged_part``), all kept in the
    work folder's scene/ and memory-mapped from there. With ``mask_volume``, the mask and what counts
    as imaged come from that volume's level, the grey values shown from the volume's."""
    m = cfg["tissue_movie"]
    root, mroot = volume_path(cfg), mask_volume_path(cfg)
    level = int(m["level"])
    out = work_dir(cfg) / "scene"
    out.mkdir(parents=True, exist_ok=True)
    path = {k: out / f"{k}_s{level}.npy" for k in ("grey", "mgrey", "mask", "shown")}
    stamps = {k: out / f"{k}_s{level}.json" for k in ("grey", "mgrey", "mask", "shown")}
    info = out / "pyramid.json"                                   # the full-resolution shape
    read = json.dumps({"volume": str(root), "level": level})
    mread = json.dumps({"volume": str(mroot), "level": level})
    made = json.dumps({"read": mread, "mask": m["mask"], "version": 9}, sort_keys=True)
    shown = {"read": read, "trim_edge_px": m["trim_edge_px"], "crop": m["crop"], "version": 2}
    shown = json.dumps(shown if mroot == root else {**shown, "mask_read": mread}, sort_keys=True)

    def current(key, text):
        return path[key].exists() and stamps[key].exists() and stamps[key].read_text() == text

    def read_level(key, volume, stamp):
        levels = open_levels(volume, 1)
        if level >= len(levels):
            raise ValueError(f"tissue_movie.level {level}: {volume.name} has levels s0-s{len(levels) - 1}")
        log.info("reading s%d of %s %s", level, volume.name, levels[level].shape)
        data = levels[level].read().result()
        if data.dtype != np.uint8:
            raise ValueError(f"{volume.name}: uint8 volumes only, not {data.dtype}")
        cli.atomic_write(path[key], lambda p: np.save(p, data))
        cli.atomic_write(stamps[key], lambda p: Path(p).write_text(stamp))
        return levels

    if not current("grey", read):
        levels = read_level("grey", root, read)
        cli.atomic_write(info, lambda p: Path(p).write_text(json.dumps({"shape0": list(levels[0].shape)})))
        stamps["mask"].unlink(missing_ok=True)
    if not info.exists():
        shape0 = list(open_levels(root, 0.1)[0].shape)
        cli.atomic_write(info, lambda p: Path(p).write_text(json.dumps({"shape0": shape0})))
    grey = np.load(path["grey"], mmap_mode="r")
    if mroot == root:
        mgrey = grey
    else:
        if not current("mgrey", mread):
            read_level("mgrey", mroot, mread)
        mgrey = np.load(path["mgrey"], mmap_mode="r")
        if mgrey.shape != grey.shape:
            raise ValueError(f"tissue_movie.mask_volume {mroot.name}: s{level} is {mgrey.shape}, "
                             f"{root.name}'s {grey.shape}: the same geometry is needed")
    src = m.get("mask_level")
    if src is not None and src < level:               # a coarser scene (a preview): the finer mask, pooled
        base = load_scene({**cfg, "tissue_movie": {**m, "level": src, "mask_level": None}}, overwrite_mask)
        made = json.dumps({"pooled": base.stamp, "level": level}, sort_keys=True)
        if not current("mask", made):
            k = 2 ** (level - src)
            fine = np.asarray(base.mask, np.float32)
            fine = np.pad(fine, [(0, k * n - v) for n, v in zip(grey.shape, fine.shape)], mode="edge")
            nz, ny, nx = grey.shape
            pooled = fine.reshape(nz, k, ny, k, nx, k).mean(axis=(1, 3, 5))
            mask = np.where(np.asarray(mgrey) > 0, np.rint(pooled), 0).astype(np.uint8)
            cli.atomic_write(path["mask"], lambda p: np.save(p, mask))
            cli.atomic_write(stamps["mask"], lambda p: Path(p).write_text(made))
    elif overwrite_mask or not current("mask", made):
        g = np.asarray(mgrey)
        log.info("tissue mask of s%d %s", level, g.shape)
        mask = tissue_mask(g, **m["mask"])
        qc = mask_qc(g, mask, qc_path(cfg, f"{movie_name(cfg)}_mask.png"))
        log.info("tissue: %.1f%% of the voxels with data; QC: %s", 100 * (mask > 0).sum() / max((g > 0).sum(), 1), qc)
        cli.atomic_write(path["mask"], lambda p: np.save(p, mask))
        cli.atomic_write(stamps["mask"], lambda p: Path(p).write_text(made))
    if not current("shown", shown):
        g = np.asarray(grey)
        if mroot != root:                              # imaged where the mask volume says (0: no data)
            g = np.where(np.asarray(mgrey) > 0, np.maximum(g, 1), 0).astype(np.uint8)
        part = crop_volume(imaged_part(g, m["trim_edge_px"]), m["crop"], 2 ** level)
        cli.atomic_write(path["shown"], lambda p: np.save(p, part))
        cli.atomic_write(stamps["shown"], lambda p: Path(p).write_text(shown))
    return Scene(np.load(path["shown"], mmap_mode="r"), np.load(path["mask"], mmap_mode="r"), 2 ** level,
                 made + shown, tuple(json.loads(info.read_text())["shape0"]))


def crop_volume(grey, crop, f):
    """``grey`` (a level with ``f`` s0 voxels per voxel) with the voxels reaching into the ``crop`` box
    (s0 voxels: any of x0, x1, y0, y1, z0, z1; what lies below x0 or above x1 etc. goes) set to 0."""
    if not crop:
        return grey
    out = np.array(grey)
    for axis, name in ((2, "x"), (1, "y"), (0, "z")):
        n = out.shape[axis]
        lo, hi = crop.get(f"{name}0"), crop.get(f"{name}1")
        index = [slice(None)] * 3
        if lo is not None and lo > 0:
            index[axis] = slice(0, min(math.ceil(lo / f), n))
            out[tuple(index)] = 0
        if hi is not None:
            index[axis] = slice(max(math.floor(hi / f), 0), n)
            out[tuple(index)] = 0
    return out


def crop_in_frame(crop, size, zoom, cx, cy):
    """1 where a frame (as ``tissue_in_frame``) shows what the ``crop`` box keeps (in x and y), else 0."""
    w, h = size
    keep = np.ones((h, w), np.float32)
    for name, c, n, axis in (("x", cx, w, 1), ("y", cy, h, 0)):
        lo, hi = crop.get(f"{name}0"), crop.get(f"{name}1")
        pos = c + (np.arange(n) + 0.5 - n / 2) * zoom
        ok = np.ones(n, bool)
        if lo is not None:
            ok &= pos >= lo
        if hi is not None:
            ok &= pos < hi
        keep *= ok[None, :] if axis == 1 else ok[:, None]
    return keep


def imaged_part(grey, trim=1):
    """``grey`` without the ``trim`` outermost voxel layers of the imaged area (set to 0, no data). The
    pyramid's mean pooling counts the zeros outside the imaged area, so a coarse voxel only partly
    imaged is darker by the share that wasn't (for 35h at s5, a median grey of 41 against 117 a voxel
    further in); as the imaged area's edge drifts by fractions of a voxel from slice to slice, those
    voxels' darkness ramps and resets every few slices, which shows as stripes on the volume's outer
    faces. The area outside is the zeros connected to the volume's faces: zeros enclosed by data (dark
    structures clipped to 0 by the display window) and the volume's own faces are not an edge."""
    if not trim:
        return grey
    lab, n = ndimage.label(grey == 0)
    outer = np.zeros(n + 1, bool)
    for face in (lab[0], lab[-1], lab[:, 0], lab[:, -1], lab[:, :, 0], lab[:, :, -1]):
        outer[np.unique(face)] = True
    outer[0] = False
    edge = ndimage.binary_dilation(outer[lab], iterations=int(trim))
    del lab
    return np.where(edge, 0, grey).astype(grey.dtype)


class Scene:
    """The level a movie is drawn from: grey (z, y, x uint8; 0: no data), the tissue mask (uint8, 255:
    tissue) and ``f`` s0 voxels per voxel; ``stamp``: what the mask was made from; ``shape0``: the
    pyramid's full-resolution shape. Arrays may be memory-mapped."""

    def __init__(self, grey, mask, f, stamp="", shape0=None):
        self.grey, self.mask, self.f, self.stamp, self.shape0 = grey, mask, f, stamp, shape0
        self.shape = grey.shape
        self._bounds = {}

    def bounds(self, axis, tissue):
        """Per slice along ``axis``: (r0, r1, c0, c1), the rows and columns holding data (or tissue);
        r0 = r1 for an empty slice."""
        key = (axis, tissue)
        if key not in self._bounds:
            m = np.asarray(self.mask) > 0 if tissue else np.asarray(self.grey) > 0
            # an image's rows and columns along axis: z -> (y, x), y -> (z, x), x -> (z, y)
            rows = {0: m.any(2), 1: m.any(2).T, 2: m.any(1).T}[axis]
            cols = {0: m.any(1), 1: m.any(0), 2: m.any(0).T}[axis]
            self._bounds[key] = np.array([_span(r) + _span(c) for r, c in zip(rows, cols)])
        return self._bounds[key]

    def box(self):
        """The volume's 8 corners (s0 voxels, x, y, z)."""
        nz, ny, nx = self.shape
        f = self.f
        return np.array([[x, y, z] for x in (0, f * nx) for y in (0, f * ny) for z in (0, f * nz)], float)


def _span(any_):
    i = np.flatnonzero(any_)
    return (int(i[0]), int(i[-1]) + 1) if len(i) else (0, 0)


# ---------------------------------------------------------------------------------------------------
# Volume rendering: slices composited back to front


def _slab(arr, axis, k):
    """Slice ``k`` of a (z, y, x) array along ``axis`` as an image: z -> rows y, columns x; y -> rows z,
    columns x; x -> rows z, columns y."""
    return arr[k] if axis == 0 else arr[:, k] if axis == 1 else arr[:, :, k]


def _quad(shape, f, axis, k):
    """(corner, du, dv) in s0 voxels of slice ``k`` along ``axis``, a level with ``f`` s0 voxels per voxel:
    the plane through the voxels' centres, the image's top-left corner and its width and height."""
    nz, ny, nx = shape
    c = f * (k + 0.5)
    if axis == 0:
        return np.array([0.0, 0.0, c]), np.array([f * nx, 0.0, 0.0]), np.array([0.0, f * ny, 0.0])
    if axis == 1:
        return np.array([0.0, c, 0.0]), np.array([f * nx, 0.0, 0.0]), np.array([0.0, 0.0, f * nz])
    return np.array([c, 0.0, 0.0]), np.array([0.0, f * ny, 0.0]), np.array([0.0, 0.0, f * nz])


def render_volume(scene, cam, size, focal, resin, cut=None, blend_axes=0.06, back=None, with_alpha=False):
    """Float frame (0-255, on black) of ``scene`` seen by ``cam`` (movie.camera): each voxel its grey
    value, tissue opaque, the rest of the data ``resin`` opaque per voxel (0: clear, 1: opaque), no data
    clear; with ``cut`` (s0 z), the volume in front of the plane z = cut is cut away (the voxel it
    passes through in part), and with ``back`` the volume behind the plane z = back. Slices along the
    axis closest to the view direction, or two stacks blended. ``with_alpha``: (frame, its opacity)."""
    forward = np.asarray(cam[1])
    order = np.argsort(-np.abs(forward))          # (x, y, z) components, most aligned first
    gap = abs(forward[order[0]]) - abs(forward[order[1]])
    pairs = [(2 - int(order[0]), 1.0)]            # (array axis, weight)
    if gap < blend_axes:
        b = 0.5 * gap / blend_axes
        pairs = [(2 - int(order[0]), 0.5 + b), (2 - int(order[1]), 0.5 - b)]
    frame, alpha = 0, 0
    for axis, weight in pairs:
        c, a = _composite(scene, axis, cam, size, focal, resin, cut, back)
        frame, alpha = frame + weight * c, alpha + weight * a
    return (frame, alpha) if with_alpha else frame


def _composite(scene, axis, cam, size, focal, resin, cut, back=None):
    w, h = size
    f, shape = scene.f, scene.shape
    pos, forward = np.asarray(cam[0]), np.asarray(cam[1])
    acc = np.zeros((h, w), np.float32)
    opacity = np.zeros((h, w), np.float32)
    n = shape[axis]
    world = 2 - axis                                  # the slices' normal in (x, y, z)
    cam_k = pos[world] / f - 0.5
    ks = sorted(range(n), key=lambda k: -abs(k - cam_k))    # back to front
    cos_axis = max(abs(forward[world]), 1e-3)
    a_resin = 1 - (1 - resin) ** (1 / cos_axis) if 0 < resin < 1 else float(resin)   # per slice, along a ray
    bounds = scene.bounds(axis, tissue=resin <= 0)
    rows, cols = shape[1:] if axis == 0 else (shape[0], shape[2]) if axis == 1 else (shape[0], shape[1])
    # the share of each voxel (along z) behind the cut and in front of the back plane
    keep = None
    if cut is not None or back is not None:
        z0 = f * np.arange(shape[0])
        keep = np.ones(shape[0], np.float32)
        if cut is not None:
            keep *= np.clip((z0 + f - cut) / f, 0, 1)
        if back is not None:
            keep *= np.clip((back - z0) / f, 0, 1)
        if not keep.any():
            return acc, opacity
    # anti-aliasing: blur by as much as the middle slice is shrunk on screen
    corner, du, dv = _quad(shape, f, axis, n // 2)
    dst = project([corner, corner + du, corner + du + dv, corner + dv], cam, size, focal)
    shrink = (2 * cols / (np.linalg.norm(dst[1] - dst[0]) + np.linalg.norm(dst[2] - dst[3])),
              2 * rows / (np.linalg.norm(dst[3] - dst[0]) + np.linalg.norm(dst[2] - dst[1])))
    sigmas = [0.5 * math.sqrt(r * r - 1) if r > 1.05 else 0.0 for r in shrink]
    pad = 2 + math.ceil(3 * max(sigmas))
    for k in ks:
        if keep is not None and axis == 0 and keep[k] <= 0:
            continue
        r0, r1, c0, c1 = bounds[k]
        if keep is not None and axis > 0:                     # rows run along z
            r0, r1 = max(r0, int(np.argmax(keep > 0))), min(r1, len(keep) - int(np.argmax(keep[::-1] > 0)))
        if r0 >= r1:
            continue
        r0, r1, c0, c1 = max(r0 - pad, 0), min(r1 + pad, rows), max(c0 - pad, 0), min(c1 + pad, cols)
        crop = (slice(r0, r1), slice(c0, c1))
        g = _slab(scene.grey, axis, k)[crop]
        shown = g > 0                                         # (no data: never shown, tissue or not)
        alpha = _slab(scene.mask, axis, k)[crop].astype(np.float32) * (1 / 255) * shown
        if a_resin > 0:
            alpha += (1 - alpha) * shown * np.float32(a_resin)
        if keep is not None:
            alpha *= keep[k] if axis == 0 else keep[r0:r1, None]
        lit = alpha * g
        if sigmas[0] or sigmas[1]:
            lit = cv2.GaussianBlur(lit, (0, 0), sigmaX=sigmas[0] or 1e-3, sigmaY=sigmas[1] or 1e-3)
            alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=sigmas[0] or 1e-3, sigmaY=sigmas[1] or 1e-3)
        corner, du, dv = _quad(shape, f, axis, k)
        corner = corner + du * (c0 / cols) + dv * (r0 / rows)
        du, dv = du * ((c1 - c0) / cols), dv * ((r1 - r0) / rows)
        quad = np.array([corner, corner + du, corner + du + dv, corner + dv])
        if ((quad - pos) @ forward).min() <= f:       # (at or behind the camera: not drawn)
            continue
        dst = project(quad, cam, size, focal)
        x0, y0 = (max(int(math.floor(v)) - 1, 0) for v in dst.min(0))
        x1, y1 = min(int(math.ceil(dst[:, 0].max())) + 1, w), min(int(math.ceil(dst[:, 1].max())) + 1, h)
        if x1 <= x0 or y1 <= y0:
            continue
        src = np.float32([[-0.5, -0.5], [c1 - c0 - 0.5, -0.5], [c1 - c0 - 0.5, r1 - r0 - 0.5], [-0.5, r1 - r0 - 0.5]])
        m = cv2.getPerspectiveTransform(src, (dst - 0.5 - [x0, y0]).astype(np.float32))
        warp = {"M": m, "dsize": (x1 - x0, y1 - y0), "flags": cv2.INTER_LINEAR, "borderMode": cv2.BORDER_CONSTANT}
        a = cv2.warpPerspective(alpha, **warp)
        view = acc[y0:y1, x0:x1]
        view *= 1 - a
        view += cv2.warpPerspective(lit, **warp)
        seen = opacity[y0:y1, x0:x1]
        seen *= 1 - a
        seen += a
    return acc, opacity


# ---------------------------------------------------------------------------------------------------
# Timeline and frames


def focal_px(m):
    """The camera's focal length in frame px: fov_deg across the frame's long side (landscape: its width;
    portrait: the same lens turned 90 degrees)."""
    return max(m["size"]) / 2 / math.tan(math.radians(m["fov_deg"]) / 2)


def fit(corners, yaw, tilt, up, size, focal, fill):
    """(look-at point, distance) of a camera turned by ``yaw`` and raised by ``tilt`` (radians; see
    pipeline.movie.camera) that frames the points ``corners``: their picture centred and ``fill`` of
    the frame's height or width, whichever is tighter."""
    w, h = size
    centre = corners.mean(0)
    span = np.ptp(corners, 0).max()
    for _ in range(4):
        lo, hi = span * 1e-2, span * 1e3
        for _ in range(48):
            mid = math.sqrt(lo * hi)
            cam = camera(centre, yaw, tilt, mid, up)
            if ((corners - cam[0]) @ cam[1]).min() <= 0:
                lo = mid
                continue
            p = project(corners, cam, size, focal)
            big = np.ptp(p[:, 0]) > fill * w or np.ptp(p[:, 1]) > fill * h
            lo, hi = (mid, hi) if big else (lo, mid)
        cam = camera(centre, yaw, tilt, hi, up)
        p = project(corners, cam, size, focal)
        off = (p.min(0) + p.max(0)) / 2 - [w / 2, h / 2]
        centre = centre + (off[0] * cam[2] + off[1] * cam[3]) * hi / focal
    return centre, hi


def _ease(t, start, end):
    """0 before ``start``, 1 after ``end``, a cosine ease in between."""
    t = np.asarray(t, float)
    u = np.clip((t - start) / (end - start), 0, 1) if end > start else (t >= start) * 1.0
    return (1 - np.cos(np.pi * u)) / 2


def _progress(t, start, end, ease):
    """0 to 1 from ``start`` to ``end``, speeding up from rest over ``ease`` and slowing to rest over the
    last ``ease`` (a cosine ramp of the speed)."""
    if end <= start:
        return (np.asarray(t) >= start) * 1.0
    e = min(ease, (end - start) / 2)
    fine = np.linspace(start, end, 4001)
    speed = _ease(fine, start, start + e) * (1 - _ease(fine, end - e, end)) if e > 0 else np.ones_like(fine)
    s = np.concatenate([[0], np.cumsum((speed[1:] + speed[:-1]) / 2 * np.diff(fine))])
    return np.interp(t, fine, s / s[-1])


def see_through(u, depth):
    """Opacity per voxel of a background ``u`` (0-1) of the way to clear: one sees ``depth * u / (1 - u)``
    voxels into it (0 at 0: opaque; ``depth`` halfway; unlimited at 1: clear)."""
    u = np.clip(np.asarray(u, float), 0, 1)
    with np.errstate(divide="ignore", over="ignore"):
        return np.where(u <= 0, 1.0, np.where(u >= 1, 0.0, -np.expm1(-(1 - u) / (depth * np.maximum(u, 1e-300)))))


def fly_settings(cfg):
    """The fly-through's ``movie:`` settings, checked against the tissue movie's."""
    m, mv = cfg["tissue_movie"], cfg["movie"]
    for key in ("size", "fps", "view_rotation_deg"):
        if list(np.atleast_1d(m[key])) != list(np.atleast_1d(mv[key])):
            raise ValueError(f"tissue_movie.fly_through: movie.{key} ({mv[key]}) must equal "
                             f"tissue_movie.{key} ({m[key]})")
    if mv["rotate_deg"]:
        raise ValueError("tissue_movie.fly_through: the fly-through has no turn (movie.rotate_deg 0)")
    return {**mv, "zoom_out_s": m["zoom_out_s"], "hold_out_s": 0.0}


def build_progress(m, scene, t, t0):
    """s0 slices built by time(s) ``t`` (``t0``: the end of the build): the first slice over first_slice_s,
    then the rest speeding up from rest and slowing to rest."""
    depth, first_s = scene.f * scene.shape[0], min(m["first_slice_s"], t0 / 2)
    return (scene.f * np.clip(np.asarray(t, float) / max(first_s, 1e-9), 0, 1)
            + (depth - scene.f) * _progress(t, first_s, t0, min(m["ease_s"], (t0 - first_s) / 2)))


def build_stop(m, scene, t0):
    """When the slicing visibly stops: after the build's last frame advancing by a voxel of the full-size
    level (``mask_level`` for a preview) or more."""
    level = m["mask_level"] if m.get("mask_level") is not None else m["level"]
    t = np.arange(0, t0 + 1e-9, 1 / m["fps"])
    if len(t) < 2:
        return float(t0)
    fast = np.flatnonzero(np.diff(build_progress(m, scene, t, t0)) >= 2 ** level)
    return float(t[fast[-1] + 1]) if len(fast) else float(t0)


def camera_path(m, t, ta, tb, end):
    """(yaw, tilt) in degrees at times ``t``: from yaw_deg[0], tilt_deg[0], drifting by drift_deg[0] until
    the turn, which runs from ``ta`` to ``tb`` (speeding up from rest over blend_s, slowing to rest over
    blend_s, a steady speed in between)
    and arrives at yaw_deg[1], tilt_deg[1]; then drifting on by drift_deg[1] (yaw) until ``end``, easing
    out. The drifts blend into the turn, so the speed changes smoothly throughout."""
    end = max(end, tb)
    fine = np.linspace(0, end, 8001)

    def walked(speed):
        c = np.concatenate([[0], np.cumsum((speed[1:] + speed[:-1]) / 2 * np.diff(fine))])
        return c / c[-1] if c[-1] > 0 else c

    out = _ease(fine, ta, tb)
    ramp = min(m.get("blend_s") or (tb - ta) / 2, (tb - ta) / 2)  # speeding up and slowing down over blend_s
    turn = walked(_ease(fine, ta, ta + ramp) * (1 - _ease(fine, tb - ramp, tb)))
    pre = walked(_ease(fine, 0, min(1.0, ta / 2) if ta > 0 else 0) * (1 - out))
    post = walked(out * (1 - _ease(fine, end - min(m["ease_s"], end - tb), end))) if end > tb else 0 * fine
    (y0, y1), (a0, a1), (d0, d1) = m["yaw_deg"], m["tilt_deg"], m["drift_deg"]
    d1 = d1 if end > tb else 0.0
    at_tb = float(np.interp(tb, fine, post))
    yaw = y0 + d0 * pre + (y1 - y0 - d0 - d1 * at_tb) * turn + d1 * post
    tilt = a0 + (a1 - a0) * turn
    return np.interp(t, fine, yaw), np.interp(t, fine, tilt)


def timeline(m, scene, fly=None):
    """One row per frame: t (s), part ("build", "orbit" or "fly"), the camera's yaw and tilt (degrees),
    look-at point lx, ly, lz and distance (s0 voxels), the background's opacity per voxel ``resin``, the
    plane z = ``cut`` in front of which the volume is cut away and z = ``back`` behind which it is (NaN:
    none), the frame's brightness ``light``, and for the fly-through (``fly``'s rows: pipeline.movie's
    zoom, cx, cy, z, rotation) the weight of the slice from the fine level on the cut, ``blend``.
    The phases: the build; the turn to the front and the resin fade (or, with fade_first, the fade then
    the turn); with fly_through, the cut sweeping from z = 0 into the fly-through's scroll as the camera
    comes face-on to it, and the fly-through. Each starts blend_s before the one before ends (after
    block_s past the build and tissue_s past the last of turn and fade), so nothing ever stands still."""
    w, h = m["size"]
    if w % 2 or h % 2:
        raise ValueError(f"tissue_movie.size {m['size']}: width and height must be even (H.264)")
    if m["view_rotation_deg"] not in (0, 180):
        raise ValueError(f"tissue_movie.view_rotation_deg must be 0 or 180, not {m['view_rotation_deg']}")
    keys = ["build_s", "block_s", "turn_s", "fade_s", "tissue_s", "fade_out_s", "ease_s", "approach_s",
            "handover_s", "blend_s"]
    if min(m[k] for k in keys) < 0 or m["turn_s"] <= 0 or m["fade_s"] <= 0:
        raise ValueError("tissue_movie: the times must be >= 0, turn_s and fade_s > 0")
    if m["blend_s"] >= min(m["turn_s"], m["fade_s"]):
        raise ValueError("tissue_movie.blend_s must be shorter than turn_s and fade_s")
    if m["fade_out_s"] > m["tissue_s"] or (m["fly_through"] and m["fade_out_s"]):
        raise ValueError("tissue_movie: fade_out_s must be <= tissue_s, and 0 with fly_through")
    if m["fly_through"] and (fly is None or m["approach_s"] <= 0):
        raise ValueError("tissue_movie.fly_through: needs the fly-through's timeline and approach_s > 0")
    fps = m["fps"]
    up = -1 if m["view_rotation_deg"] == 0 else 1
    focal = focal_px(m)
    ov = m["blend_s"]
    t0 = m["build_s"]                                   # the stack is complete
    first = max(t0 + m["block_s"] - ov, 0.0)            # the next phase starts
    if m["fade_first"]:
        f0 = first
        ta = f0 + m["fade_s"] - ov
    else:
        ta = first
        if m["fade_at_yaw_deg"] is not None:          # as the turning camera passes this yaw
            tt = np.arange(0, ta + m["turn_s"] + 1e-9, 1 / fps)
            yy = camera_path(m, tt, ta, ta + m["turn_s"], ta + m["turn_s"])[0] - m["fade_at_yaw_deg"]
            past = np.sign(m["yaw_deg"][1] - m["yaw_deg"][0]) * yy >= 0
            f0 = float(tt[int(np.argmax(past))]) if past.any() else ta + m["turn_s"]
        elif m["fade_start_s"] is not None:           # after the slicing stops
            f0 = max(build_stop(m, scene, t0) + m["fade_start_s"], 0.0)
        else:
            f0 = ta + m["turn_s"] - ov
    tb, f1 = ta + m["turn_s"], f0 + m["fade_s"]         # the turn and the fade end
    if m["fly_through"]:       # the sweep starts blend_s before the turn ends (the fade finishing during it)
        t3 = max(tb + m["tissue_s"] - ov, f1 - m["approach_s"])
    else:
        t3 = max(tb, f1) + m["tissue_s"]                # the end
    t4 = t3 + (m["approach_s"] if m["fly_through"] else 0)              # the fly-through takes over
    n = int(round(t4 * fps))
    t = np.arange(n) / fps
    yaw, tilt = camera_path(m, t, ta, tb, t3)
    corners = scene.box()
    poses = [fit(corners, math.radians(a), math.radians(b), up, (w, h), focal, m["fill"]) for a, b in zip(yaw, tilt)]
    look = np.array([p[0] for p in poses]).reshape(n, 3)
    dist = np.log([p[1] for p in poses])
    if n > 1:      # smoothed over half a second: the outermost corners change as the camera turns
        look = ndimage.gaussian_filter1d(look, 0.5 * fps, axis=0, mode="nearest")
        dist = ndimage.gaussian_filter1d(dist, 0.5 * fps, mode="nearest")
    dist = np.exp(dist) * (1 - m["push_in"] * _ease(t, min(f0, ta), t4 if m["fly_through"] else t3))
    # The background: how deep one sees into it grows smoothly from 0 to unlimited, resin_depth voxels
    # halfway (resin_depth * u / (1 - u)).
    resin = see_through(_ease(t, f0, f1), float(m["resin_depth"]))
    # The build: the first slice over first_slice_s, then the rest speeding up from rest and slowing to
    # rest; from the last slice to the first (the volume behind the cut), or as imaged (in front of back).
    depth = scene.f * scene.shape[0]
    built = build_progress(m, scene, t, t0)
    if m["build_order"] not in ("reverse", "imaging"):
        raise ValueError(f"tissue_movie.build_order must be reverse or imaging, not {m['build_order']!r}")
    building = t < t0
    cut = np.where(building & (m["build_order"] == "reverse"), depth - np.minimum(built, depth), np.nan)
    back = np.where(building & (m["build_order"] == "imaging"), np.minimum(built, depth), np.nan)
    if m["fly_through"]:
        # The camera comes face-on to the cut, which sweeps from z = 0 to the fly-through's first slice,
        # arriving there at its scroll speed: speeding up from rest, then slowing into the scroll.
        # The zoom-in starts zoom_overlap_s before the hand-over: the approach blends into the fly-through's
        # moving camera (its zoom, centre and cut), arriving exactly in its state and moving with it.
        ov_fly = min(m["zoom_overlap_s"], float(fly["t"].iloc[-1]))
        tf = np.clip(t - (t4 - ov_fly), 0, None)
        zoomf, cxf, cyf = (np.interp(tf, fly["t"], fly[k]) for k in ("zoom", "cx", "cy"))
        z_end = float(np.interp(ov_fly, fly["t"], fly["z"]))
        v_end = float(np.interp(ov_fly + 0.5 / fps, fly["t"], fly["z"]) - np.interp(ov_fly - 0.5 / fps, fly["t"], fly["z"])) * fps
        sweep = sweep_path(t, t3, t4, z_end + 0.5, v_end)
        e = _ease(t, t3, t4)
        yaw, tilt = yaw * (1 - e), tilt * (1 - e)
        target = np.stack([cxf, cyf, np.where(np.isfinite(sweep), sweep, z_end + 0.5)], -1)
        look = look * (1 - e[:, None]) + target * e[:, None]
        dist = np.exp(np.log(dist) * (1 - e) + np.log(focal * zoomf) * e)
        fly = fly[fly["t"] >= ov_fly - 1e-9].copy()   # (the rest of the fly-through, from the hand-over)
        fly["t"] = fly["t"] - fly["t"].iloc[0]
        fly["frame"] = np.arange(len(fly))
        cut = np.where(np.isfinite(sweep), sweep, cut)
    light = 1 - _ease(t, t3 - m["fade_out_s"], t3) if m["fade_out_s"] else np.ones(n)
    plan = pd.DataFrame({"frame": np.arange(n), "t": t, "part": np.where(t < t0 - 1e-9, "build", "orbit"),
                         "yaw": yaw, "tilt": tilt, "lx": look[:, 0], "ly": look[:, 1], "lz": look[:, 2], "dist": dist,
                         "resin": resin, "cut": cut, "back": back, "light": light})
    if m["fly_through"]:
        plan = pd.concat([plan, _fly_rows(m, scene, fly, n, t4, focal, up)], ignore_index=True)
    return plan


def _fly_rows(m, scene, fly, n0, t0, focal, up):
    """The fly-through's rows from frame ``n0`` (time ``t0``): its zoom, cx, cy, z, rotation and ``blend``,
    the fine slice's weight on the cut (in over handover_s, out again before the outro). With a zoom-out,
    the scroll reverses as it starts, within back_turn_s, and runs back at back_rate slices/s per zoom
    (voxels per frame px on a frame ref_width px along its long side: the same speed on screen), and faster
    still when far out (x (1 + (zoom /
    back_boost_zoom)^2)), the removed slices
    reappearing on the face towards the camera. zoom_out_cut of the way through the zoom-out the outro
    takes over: the camera, keeping its speed, pulls back face-on to the whole tissue volume (over at
    least pull_min_s) while the scroll-back goes on, its last slice landing by the time the pull-back ends;
    the camera then turns once round the volume (spin_s, starting spin_overlap_s before that), easing to
    rest, and the frame fades to black over end_fade_s."""
    tf = fly["t"].to_numpy()
    rows = fly[["zoom", "cx", "cy", "z", "rotation"]].assign(frame=fly["frame"] + n0, t=tf + t0, part="fly")
    rows["blend"] = _ease(tf, 0, m["handover_s"])
    if not m["zoom_out_s"]:
        return rows
    fps, (w, h) = m["fps"], m["size"]
    boost = m["back_boost_zoom"]

    def back_speed(zoom):                             # slices/s at ``zoom`` (this frame size's)
        full = zoom * max(w, h) / m["ref_width"]      # (the zoom on a frame ref_width px along its long side)
        return m["back_rate"] * full * (1 + (full / boost) ** 2 if boost else 1)

    zo0 = tf[-1] + 1 / fps - m["zoom_out_s"]          # the zoom-out ends with the fly-through
    tj = zo0 + m["zoom_out_cut"] * m["zoom_out_s"]    # the outro takes over
    keep = tf < tj - 1e-9
    j = int(keep.sum())                               # the first outro frame
    # The scroll-back in the fly-through: from the forward scroll to rate x zoom backwards within
    # back_turn_s (the cut plane at z + 0.5)
    i0 = int(np.searchsorted(tf, zo0))
    z = rows["z"].to_numpy().copy()
    va = (z[i0] - z[i0 - 1]) * fps
    zoom = rows["zoom"].to_numpy()
    for k in range(i0 + 1, len(z)):
        e = float(_ease(tf[k], zo0, zo0 + m["back_turn_s"]))
        z[k] = max(z[k - 1] + (va * (1 - e) - back_speed(zoom[k]) * e) / fps, -0.5)
    rows["z"] = z
    rows["blend"] = rows["blend"] * (1 - _ease(tf, tj - 0.3 * (tj - zo0), tj))
    rows = rows[keep]
    # The outro. The camera, face-on (yaw 0) at the hand-over, moves on from where it is (cx, cy, z + 0.5 -
    # focal x zoom) at its speed, easing into the whole volume's face-on framing over pull_s; the cut goes
    # on back as fast as at the zoom it is seen at (its distance from the camera / focal), slowing to rest
    # over its last stretch, and
    # lands as the pull-back ends (pull_s solved for that).
    cam = np.c_[fly["cx"], fly["cy"], z + 0.5 - focal * zoom]
    c0, v0 = cam[j], (cam[j] - cam[j - 1]) * fps
    look_f, dist_f = fit(scene.box(), 0.0, 0.0, up, (w, h), focal, m["fill"])
    c1 = look_f - dist_f * np.array([0.0, 0.0, 1.0])
    cut0, vcut0 = z[j] + 0.5, (z[j] - z[j - 1]) * fps

    def path(pull_s, n):
        t = np.arange(n) / fps
        u = np.clip(t / pull_s, 0, 1)[:, None]        # cubic Hermite: from (c0, v0) to (c1, at rest)
        c = ((2 * u ** 3 - 3 * u ** 2 + 1) * c0 + (u ** 3 - 2 * u ** 2 + u) * pull_s * v0
             + (-2 * u ** 3 + 3 * u ** 2) * c1)
        cut = np.empty(n)
        zc, v = cut0, vcut0
        for k in range(n):
            if k:
                zc = max(zc + v / fps, 0.0)
            cut[k] = zc
            speed = back_speed(max(zc - c[k, 2], 0.0) / focal)
            land = math.sqrt(zc / (speed * m["back_land_s"])) if speed > 0 else 1.0
            v = -speed * min(1.0, land) if zc > 0 else 0.0
        return t, c, cut

    pull = 3.0
    for _ in range(8):                                # the pull-back ends as the last slice lands
        t, c, cut = path(pull, int(90 * fps))
        pull = max(float(t[int(np.argmax(cut <= 0))]) if (cut <= 0).any() else float(t[-1]), m["pull_min_s"])
    spin0 = max(pull - m["spin_overlap_s"], 0.0)
    n = int(round((spin0 + m["spin_s"] + m["end_fade_s"] / 2) * fps))
    t, c, cut = path(pull, n)
    fine = np.linspace(0, t[-1] + 1, 8001)            # the spin: speeding up over spin_in_s, easing out over the rest
    speed = _ease(fine, spin0, spin0 + m["spin_in_s"]) * (1 - _ease(fine, spin0 + m["spin_in_s"], spin0 + m["spin_s"]))
    walked = np.concatenate([[0], np.cumsum((speed[1:] + speed[:-1]) / 2 * np.diff(fine))])
    spin = np.radians(m["spin_deg"]) * np.interp(t, fine, walked / walked[-1])
    tilt = np.radians(m["spin_tilt_deg"]) * _ease(t, spin0, spin0 + m["spin_s"] / 2)
    poses = [fit(scene.box(), a, b, up, (w, h), focal, m["fill"]) for a, b in zip(spin, tilt)]
    look = ndimage.gaussian_filter1d(np.array([p[0] for p in poses]), 0.5 * fps, axis=0, mode="nearest")
    dist = np.exp(ndimage.gaussian_filter1d(np.log([p[1] for p in poses]), 0.5 * fps, mode="nearest"))
    look = look + (c - c1)                            # the pull-back moves camera and look-at point together
    outro = pd.DataFrame({"frame": n0 + j + np.arange(n), "t": j / fps + t0 + t, "part": "orbit",
                          "yaw": np.degrees(spin), "tilt": np.degrees(tilt), "lx": look[:, 0], "ly": look[:, 1],
                          "lz": look[:, 2], "dist": dist, "resin": 0.0, "cut": np.where(cut > 0, cut, np.nan),
                          "back": np.nan, "light": 1 - _ease(t, t[-1] - m["end_fade_s"], t[-1] + 1 / fps)})
    return pd.concat([rows, outro], ignore_index=True)


def sweep_path(t, start, end, z_end, v_end):
    """The cut's z (s0 voxels; NaN outside ``start``-``end``): from 0 at rest to ``z_end``, arriving at
    speed ``v_end`` (s0 slices/s) with no acceleration: v = v_end (1 - cos)/2 + A sin^2 over the sweep."""
    u = np.clip((np.asarray(t, float) - start) / (end - start), 0, 1)
    span = end - start
    a = 2 * z_end / span - v_end                      # the bump's height, from the distance covered
    z = span * (v_end * (u - np.sin(np.pi * u) / np.pi) / 2 + a * (u - np.sin(2 * np.pi * u) / (2 * np.pi)) / 2)
    return np.where((np.asarray(t) >= start) & (np.asarray(t) <= end), z, np.nan)


def fly_plan(cfg, shape0):
    """The fly-through's timeline: pipeline.movie.timeline of the ``movie:`` settings without the hold at
    the start (the zoom starts as the cut's sweep ends: see zoom_overlap_s)."""
    return movie.timeline({**fly_settings(cfg), "hold_start_s": 0.0}, shape0)


def render_row(scene, m, row):
    """Float frame (unrotated) of a timeline row of part "build" or "orbit"."""
    size = m["size"]
    up = -1 if m["view_rotation_deg"] == 0 else 1
    focal = focal_px(m)
    cam = camera((row.lx, row.ly, row.lz), math.radians(row.yaw), math.radians(row.tilt), row.dist, up)
    cut = None if not np.isfinite(row.cut) else row.cut
    back = None if not np.isfinite(row.back) else row.back
    return render_volume(scene, cam, size, focal, row.resin, cut, m["blend_axes"], back) * row.light


def tissue_in_frame(scene, size, zoom, cx, cy, z):
    """The tissue mask (float 0-1) on a fly-through frame of slice ``z`` around (cx, cy) at ``zoom`` s0
    voxels per frame px (unrotated, as pipeline.movie.render_frame): the scene's mask, interpolated
    linearly in x, y and z, so its edges are as sharp as the mask's voxels allow."""
    w, h = size
    f = scene.f
    r = zoom / f                                      # mask voxels per frame px
    u_first = (cx + (0.5 - w / 2) * zoom) / f - 0.5
    v_first = (cy + (0.5 - h / 2) * zoom) / f - 0.5
    cl = min(max((z + 0.5) / f - 0.5, 0.0), scene.shape[0] - 1.0)
    ca = int(math.floor(cl))
    frac = cl - ca
    pad = 2
    u0, v0 = math.floor(u_first) - pad, math.floor(v_first) - pad
    u1, v1 = math.ceil(u_first + (w - 1) * r) + pad + 1, math.ceil(v_first + (h - 1) * r) + pad + 1
    nz, ny, nx = scene.shape
    plane = np.zeros((v1 - v0, u1 - u0), np.float32)
    ya, yb, xa, xb = max(v0, 0), min(v1, ny), max(u0, 0), min(u1, nx)
    if ya < yb and xa < xb:
        sub = scene.mask[ca, ya:yb, xa:xb].astype(np.float32)
        if frac > 1e-3 and ca + 1 < nz:
            sub = sub * (1 - frac) + scene.mask[ca + 1, ya:yb, xa:xb].astype(np.float32) * frac
        plane[ya - v0:yb - v0, xa - u0:xb - u0] = sub * (1 / 255)
    m = np.array([[r, 0, u_first - u0], [0, r, v_first - v0]], np.float64)
    return cv2.warpAffine(plane, m, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def refine_edge(tissue, frame, split=0.5, min_contrast=20):
    """The tissue mask on a frame (``tissue_in_frame``) with its edge placed by the frame itself: within
    the mask's edge band (one mask voxel wide), a pixel is resin if it is brighter than ``split`` of the
    way from the frame's tissue grey to its resin grey (the medians where the mask is sure), as the
    mask's own rim criterion; a soft step over a tenth of that span. Unchanged where the frame shows
    no resin, or no contrast."""
    band = (tissue > 1e-3) & (tissue < 1 - 1e-3)
    resin = (tissue < 1e-3) & (frame > 0)
    if not band.any() or resin.sum() < 100 or (tissue > 1 - 1e-3).sum() < 100:
        return tissue
    t, r = float(np.median(frame[tissue > 1 - 1e-3])), float(np.median(frame[resin]))
    if r - t < min_contrast:
        return tissue
    g = cv2.GaussianBlur(frame.astype(np.float32), (0, 0), 1.5)
    level = (g - t) / (r - t)
    sharp = np.clip((split - level) / 0.1 + 0.5, 0, 1)
    return np.where(band, sharp, tissue).astype(np.float32)


def frames_dir(cfg):
    scale = cfg["tissue_movie"]["preview_scale"]
    if scale == 1:
        return work_dir(cfg)
    out = work_dir(cfg) / f"preview{scale}"
    out.mkdir(exist_ok=True)
    return out


def preview(cfg, scale):
    """``cfg`` for a preview ``scale`` times smaller: the frame size divided by it and every zoom (s0
    voxels per frame px) multiplied by it, so the frames show the same as the full-size ones (from
    coarser levels: the volume too, from the level log2(scale) coarser, its tissue mask pooled from the
    mask of the full-size level); its frames and movie are kept apart (``frames_dir``, ``encode_command``)."""
    scale = int(scale)
    if scale < 1 or any(v % (2 * scale) for v in cfg["tissue_movie"]["size"]):
        raise ValueError(f"preview {scale}: the frame size divided by it must stay even")
    out = deep_merge(cfg, {})
    m, mv = out["tissue_movie"], out["movie"]
    m["preview_scale"] = scale
    m["mask_level"], m["level"] = m["level"], m["level"] + int(math.log2(scale))   # the volume from a coarser
    m["size"] = [v // scale for v in m["size"]]                                    # level, its mask pooled
    mv["size"] = [v // scale for v in mv["size"]]
    for key in ("zoom_end", "zoom_start", "zoom_out_to", "rotate_start_zoom"):
        if mv.get(key) is not None:
            mv[key] = mv[key] * scale
    return out


_WORKER = {}


def _state(cfg, levels=False):
    """The scene (and, with ``levels``, the fly-through's pyramid), once per process."""
    if "scene" not in _WORKER:
        _WORKER["scene"] = load_scene(cfg)
    if levels and "levels" not in _WORKER:
        _WORKER["levels"] = open_levels(movie.volume_path(cfg), cfg["movie"]["cache_gb"])
        mroot = mask_volume_path(cfg)
        _WORKER["mask_levels"] = (open_levels(mroot, cfg["movie"]["cache_gb"])
                                  if mroot != volume_path(cfg) else None)
    return _WORKER


def _init_worker(cfg, threads):
    cv2.setNumThreads(threads)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    _WORKER["cfg"] = cfg
    _state(cfg)


def render_frame(cfg, row):
    """Frame (uint8, as shown: rotated by view_rotation_deg) of a timeline row (a namespace)."""
    m = cfg["tissue_movie"]
    state = _state(cfg, levels=row.part == "fly")
    if row.part == "fly":
        frame = fly_picture(cfg, state, row.zoom, row.cx, row.cy, row.z, row.rotation, row.blend)
    else:
        frame = render_row(state["scene"], m, row)
    frame = np.clip(np.rint(frame), 0, 255).astype(np.uint8)
    frame = np.ascontiguousarray(np.rot90(frame, 2)) if m["view_rotation_deg"] == 180 else frame
    if row.part == "fly" and cfg["movie"]["scale_bar"]:
        frame = movie.draw_scale_bar(frame, row.zoom * (serve.voxel_nm(movie.volume_path(cfg)) or {"x": 8.0})["x"])
    return frame


def fly_picture(cfg, state, zoom, cx, cy, z, rotation=0.0, fine_weight=1.0):
    """A fly-through frame (float, unrotated), tissue only and in 3D: the camera face-on to slice ``z``
    at the distance that shows it at ``zoom`` s0 voxels per frame px around (cx, cy), the volume in front
    cut away there; on the cut, the slice from the level the zoom needs (pipeline.movie.render_frame),
    its tissue only (``refine_edge`` of ``tissue_in_frame``), ``fine_weight`` of it over the scene's own
    picture of the cut (the hand-over from the scene's level); through the resin, the tissue behind it,
    from the scene."""
    m = cfg["tissue_movie"]
    size = m["size"]
    focal = focal_px(m)
    up = -1 if m["view_rotation_deg"] == 0 else 1
    cam = camera((cx, cy, z + 0.5), 0.0, 0.0, focal * zoom, up)
    picture = render_volume(state["scene"], cam, size, focal, 0.0, z + 0.5, m["blend_axes"])
    if fine_weight > 0:
        levels = _state(cfg, levels=True)["levels"]
        fine = movie.render_frame(levels, size, zoom, cx, cy, z, rotation)[0].astype(np.float32)
        guide = fine                                  # (the edge placed from the mask volume's greys)
        if state.get("mask_levels") is not None:
            guide = movie.render_frame(state["mask_levels"], size, zoom, cx, cy, z, rotation)[0].astype(np.float32)
        tissue = refine_edge(tissue_in_frame(state["scene"], size, zoom, cx, cy, z), guide) * np.float32(fine_weight)
        if m["crop"]:
            tissue = tissue * crop_in_frame(m["crop"], size, zoom, cx, cy)
        picture = tissue * fine + (1 - tissue) * picture
    return picture


def _write_frame(row, cfg=None):
    row = SimpleNamespace(**row)            # a timeline row as a dict (namedtuples don't pickle)
    cfg = cfg or _WORKER["cfg"]
    frame = render_frame(cfg, row)
    path = frames_dir(cfg) / f"f{row.frame:05d}.png"
    cli.atomic_write(path, lambda p: cv2.imwrite(p, frame, [cv2.IMWRITE_PNG_COMPRESSION, 1]))
    return row.frame


def plan_of(cfg):
    """(Scene, timeline) of a config."""
    m = cfg["tissue_movie"]
    state = _state(cfg)
    fly = fly_plan(cfg, state["scene"].shape0) if m["fly_through"] else None
    return state["scene"], timeline(m, state["scene"], fly)


def _reusable(old_text, new_text):
    """{new frame: old frame} for the rows of a new timeline.csv text that an old one has too (all but the
    frame number and time the same; a missing column counts as empty); none if their settings lines
    differ."""
    if not old_text or old_text.split("\n", 1)[0] != new_text.split("\n", 1)[0]:
        return {}
    import io
    old, new = (pd.read_csv(io.StringIO(text), comment="#", dtype=str, keep_default_na=False)
                for text in (old_text, new_text))
    cols = sorted((set(old.columns) | set(new.columns)) - {"frame", "t"})
    old, new = old.reindex(columns=["frame", *cols], fill_value=""), new.reindex(columns=["frame", *cols], fill_value="")
    found = {}
    for row in old.itertuples(index=False):
        found.setdefault(tuple(row[1:]), int(row[0]))
    return {int(row[0]): found[tuple(row[1:])] for row in new.itertuples(index=False) if tuple(row[1:]) in found}


def make_frames(cfg, only=None, overwrite=False, workers=None):
    """Write the PNG frames (all, or the frame numbers in ``only``); frames from an earlier run with the
    same look and the same timeline row (if at another time) are kept, the others redone."""
    m = cfg["tissue_movie"]
    _WORKER.clear()
    scene, plan = plan_of(cfg)
    out = frames_dir(cfg)
    plan_csv = out / "timeline.csv"
    keys = ("size", "view_rotation_deg", "level", "fov_deg", "blend_axes")
    settings = {k: m[k] for k in keys} | {"volume": str(volume_path(cfg)), "mask": scene.stamp}
    if m["fly_through"]:
        settings["fly_volume"] = str(movie.volume_path(cfg))
        settings["scale_bar"] = cfg["movie"]["scale_bar"]
    plan_text = f"# {json.dumps(settings, sort_keys=True)}\n" + plan.to_csv(index=False, float_format="%.6f")
    if overwrite or not plan_csv.exists() or plan_csv.read_text() != plan_text:
        old_text = "" if overwrite or not plan_csv.exists() else plan_csv.read_text()
        reuse = {new: old for new, old in _reusable(old_text, plan_text).items()
                 if (out / f"f{old:05d}.png").exists()}
        keep = out / "reuse"                          # frames still wanted, under their old numbers
        keep.mkdir(exist_ok=True)
        for old in set(reuse.values()):
            os.replace(out / f"f{old:05d}.png", keep / f"f{old:05d}.png")
        for stale in out.glob("f*.png"):
            stale.unlink()
        for new, old in reuse.items():
            os.link(keep / f"f{old:05d}.png", out / f"f{new:05d}.png")
        shutil.rmtree(keep)
        if reuse:
            log.info("%d frames kept from the earlier run", len(reuse))
        cli.atomic_write(plan_csv, lambda p: Path(p).write_text(plan_text))
    rows = plan if only is None else plan[plan["frame"].isin(only)]
    todo = [row for row in rows.to_dict("records") if not (out / f"f{row['frame']:05d}.png").exists()]
    workers = max(1, int(workers or m["workers"]))
    log.info("%s: %d of %d frames to render (%dx%d, %.1f s at %d fps) from s%d %s, %d worker(s)",
             movie_name(cfg), len(todo), len(plan), *m["size"], len(plan) / m["fps"], m["fps"], m["level"],
             scene.shape, workers)
    done = 0
    if workers == 1 or len(todo) <= 1:
        _WORKER["cfg"] = cfg
        for row in todo:
            _write_frame(row, cfg)
            done += 1
            if done % 25 == 0:
                log.info("frame %d (%d/%d)", row["frame"], done, len(todo))
    else:
        import multiprocessing
        threads = max(1, (os.cpu_count() or 1) // workers)
        with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_init_worker, initargs=(cfg, threads)) as pool:
            for frame in pool.map(_write_frame, todo, chunksize=1):
                done += 1
                if done % 25 == 0:
                    log.info("frame %d (%d/%d)", frame, done, len(todo))
    log.info("%d frames written to %s", done, out)
    return out


def encode_command(cfg):
    m = cfg["tissue_movie"]
    scale = m["preview_scale"]
    out = Path(cfg["output_dir"]) / f"{movie_name(cfg)}{f'_preview{scale}' if scale > 1 else ''}.mp4"
    return [m["ffmpeg"], "-y", "-loglevel", "warning", "-framerate", str(m["fps"]),
            "-i", str(frames_dir(cfg) / "f%05d.png"), "-c:v", "libx264", "-preset", "slow", "-crf", str(m["crf"]),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]


def encode(cfg):
    plan_csv = frames_dir(cfg) / "timeline.csv"
    if not plan_csv.exists():
        raise RuntimeError("no timeline: run 'frames' first")
    n = len(pd.read_csv(plan_csv, comment="#"))
    missing = [k for k in range(n) if not (frames_dir(cfg) / f"f{k:05d}.png").exists()]
    if missing:
        raise RuntimeError(f"{len(missing)} of {n} frames missing (first: {missing[0]}): run 'frames' first")
    cmd = encode_command(cfg)
    subprocess.run(cmd, check=True)
    log.info("wrote %s", cmd[-1])


def main(argv=None):
    p = cli.base_parser(__doc__.split("\n\n")[0])
    p.add_argument("action", choices=("mask", "frames", "encode"))
    p.add_argument("--frames", help="frames: only these frame numbers, e.g. 0,240,480 (a quick look)")
    p.add_argument("--workers", type=int, help="frames: parallel processes (default: tissue_movie.workers)")
    p.add_argument("--preview", type=int, help="frames, encode: a preview this many times smaller (default: "
                   "tissue_movie.preview_scale)")
    p.add_argument("--print", action="store_true", help="encode: print the ffmpeg command instead of running it")
    args = p.parse_args(argv)
    cfg = cli.setup(args, deep_merge(movie.DEFAULTS, DEFAULTS))
    scale = args.preview or cfg["tissue_movie"]["preview_scale"]
    if scale > 1:
        cfg = preview({**cfg, "tissue_movie": {**cfg["tissue_movie"], "preview_scale": 1}}, scale)
    if args.action == "mask":
        load_scene(cfg, overwrite_mask=args.overwrite)
    elif args.action == "frames":
        only = [int(v) for v in args.frames.split(",")] if args.frames else None
        make_frames(cfg, only, args.overwrite, args.workers)
    elif args.print:
        print(shlex.join(encode_command(cfg)))
    else:
        if not shutil.which(cfg["tissue_movie"]["ffmpeg"]):
            raise FileNotFoundError(f"{cfg['tissue_movie']['ffmpeg']} not found (on NEMO: ml FFmpeg)")
        encode(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
