"""Fly-through movie of a rendered volume: the whole slice first, then a smooth zoom into a target
point down to full resolution, scrolling through z the whole time (faster when zoomed out, if set);
optionally the view then turns 90 degrees about the frame's vertical axis to the y-z plane (showing
the voxels are isotropic), and finally zooms back out until the whole plane (e.g. the side view, y
by z) fills the frame; with rotate_start_zoom the turn comes during the zoom-out instead, once it
is zoomed out that far. The turn shows the cutting plane turning face-on about the frame's vertical
centre line, the scroll carrying on along its normal; or, with rotate_style "block", the field of view
as a solid block in perspective (the scroll pauses, the camera pulls back and rises a little so three
faces show, and comes to rest facing the block's side, the y-z plane).

Each frame reads only what is on screen, from the pyramid level just finer than the frame's zoom
(the whole slice at 64-128 nm at the start, a frame-sized window at 8 nm at the end), so a 20 s
movie reads a few GB whatever the size of the volume. The zoom eases in and out and runs at a
constant rate in log scale; the view centre moves so the target stays on the straight line from
its place in the overview to the frame centre. Slices between two of a coarse level's planes are
blended linearly, so scrolling stays smooth when zoomed out.

    python -m pipeline.movie --config C frames     # PNG frames -> work/movie/<name>/
    python -m pipeline.movie --config C encode     # ffmpeg -> output_dir/<name>.mp4
    sbatch slurm/movie.sbatch C                    # both, on NEMO

Coordinates (``movie.target``) are full-resolution voxels as neuroglancer shows them for the
volume (x, y, z).
"""

import logging
import math
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorstore as ts
from scipy import ndimage

from . import cli, serve
from .config import step_dir

log = logging.getLogger(__name__)

DEFAULTS = {
    "movie": {
        "volume": None,           # OME-Zarr in output_dir to film; null: <name>.ome.zarr
        "name": None,             # -> output_dir/<name>.mp4; null: <volume stem>_movie
        "target": None,           # [x, y, z] full-resolution voxels: the point zoomed into; z is the
                                  # slice on screen when the zoom ends
        "size": [1920, 1080],     # frame width, height (px); [3840, 2160] for 4K
        "view_rotation_deg": 0,   # rotates the picture in the frame plane: 0 or 180
        "fps": 30,
        "slices_per_s": 30,       # scroll speed through z in full-resolution slices; or [zoomed out, zoomed in],
                                  # e.g. [120, 60], changing with the zoom
        "direction": 1,           # 1: towards higher z, -1: towards lower z
        "hold_start_s": 2.0,      # on the whole slice before the zoom starts
        "zoom_s": 14.0,           # the zoom
        "hold_end_s": 4.0,        # at the final zoom after it
        "rotate_deg": 0,          # then turn the cutting plane about the frame's vertical axis by this much
                                  # (90: to the y-z plane); 0: no rotation
        "rotate_s": 6.0,          # the rotation
        "rotated_s": 4.0,         # after it
        "rotate_start_zoom": None,  # or turn during the zoom-out, from when it reaches this zoom (voxels per
                                    # frame px), over rotate_s; no rotated_s (plane, rotate_deg +-90)
        "rotate_style": "plane",  # plane: the cutting plane turns face-on and the scroll carries on along its
                                  # normal; block: the field of view turns as a solid block (as deep as the
                                  # frame is wide), seen in perspective, the scroll paused
        "rotate_pause_s": 1.0,    # block: the scroll eases to a stop over this long before the turn, and
                                  # back up after it
        "rotate_fov_deg": 40,     # block: the camera's horizontal field of view (more: stronger perspective)
        "rotate_tilt_deg": 20,    # block: mid-turn the camera rises this much, so the top face shows too
        "rotate_pull_back": None, # block: mid-turn camera distance / face-on distance; null: the whole block
                                  # fits the frame
        "zoom_out_s": 0.0,        # then zoom back out over this long (0: no zoom-out); rotate_deg 0 or +-90
        "zoom_out_to": None,      # voxels per frame px at the end; null: the whole final plane fits the frame
        "hold_out_s": 3.0,        # after the zoom-out
        "zoom_start": None,       # voxels per frame px at the start; null: the whole slice fits the frame
        "zoom_end": 1.0,          # voxels per frame px at the end (1: full resolution; 0.5: 2x magnified)
        "scale_bar": False,
        "crf": 18,                # x264 quality: lower is better and bigger (18: visually lossless)
        "ffmpeg": "ffmpeg",
        "cache_gb": 4,            # decoded chunks kept in memory between frames
    },
}


def volume_path(cfg):
    m = cfg["movie"]
    return Path(cfg["output_dir"]) / (m["volume"] or f"{cfg['name']}.ome.zarr")


def movie_name(cfg):
    m = cfg["movie"]
    return m["name"] or volume_path(cfg).name.removesuffix(".ome.zarr") + "_movie"


def _ease(t, start, end):
    """0 before ``start``, 1 after ``end``, a cosine ease in between."""
    u = np.clip((t - start) / (end - start), 0, 1) if end > start else (t >= start).astype(float)
    return (1 - np.cos(np.pi * u)) / 2


def _integral(v, t, anchor):
    """Integral of ``v`` (sampled at frame times ``t``) over time, 0 at time ``anchor``."""
    c = np.concatenate([[0], np.cumsum((v[1:] + v[:-1]) / 2 * np.diff(t))]) if len(t) > 1 else np.zeros(1)
    at = np.interp(anchor, t, c) if anchor <= t[-1] else c[-1] + v[-1] * (anchor - t[-1])
    return c - at


def timeline(m, shape0):
    """One row per frame: t (s), zoom (s0 voxels per frame px), the view centre cx, cy, z (s0 voxels;
    z = an exact slice), the view's rotation about the frame's vertical axis (degrees) and how far
    through the turn it is (0-1; a block turn renders the frames strictly between 0 and 1)."""
    if not m["target"] or len(m["target"]) != 3:
        raise ValueError("set movie.target: [x, y, z] in voxels, as neuroglancer shows them")
    w, h = m["size"]
    if w % 2 or h % 2:
        raise ValueError(f"movie.size {m['size']}: width and height must be even (H.264)")
    if m["view_rotation_deg"] not in (0, 180):
        raise ValueError(f"movie.view_rotation_deg must be 0 or 180, not {m['view_rotation_deg']}")
    fps = m["fps"]
    t0 = m["hold_start_s"]
    t1 = t0 + m["zoom_s"]
    t2 = t1 + m["hold_end_s"]
    during = bool(m["rotate_deg"]) and m.get("rotate_start_zoom") is not None     # the turn during the zoom-out
    t3 = t2 + (m["rotate_s"] if m["rotate_deg"] and not during else 0)
    t4 = t3 + (m["rotated_s"] if m["rotate_deg"] and not during else 0)
    t5 = t4 + m["zoom_out_s"]
    if min(t0, m["hold_end_s"], m["rotate_s"], m["rotated_s"], m["zoom_out_s"], m["hold_out_s"]) < 0 \
            or m["zoom_s"] <= 0:
        raise ValueError("movie: the hold, rotation and zoom-out times must be >= 0 and zoom_s > 0")
    side = abs(abs(m["rotate_deg"]) - 90) < 1e-9
    if m["zoom_out_s"] and not (side or m["rotate_deg"] == 0):
        raise ValueError("movie: zoom_out_s needs rotate_deg 0, 90 or -90")
    if m["rotate_style"] not in ("block", "plane"):
        raise ValueError(f"movie.rotate_style must be block or plane, not {m['rotate_style']!r}")
    block = bool(m["rotate_deg"]) and m["rotate_style"] == "block"
    if block:
        fov, tilt, k = m["rotate_fov_deg"], m["rotate_tilt_deg"], m["rotate_pull_back"]
        if not side:
            raise ValueError("movie: rotate_style block needs rotate_deg 90 or -90")
        if m["rotate_pause_s"] < 0 or not 0 < fov < 180 or not abs(tilt) < 90 or (k is not None and k <= 0):
            raise ValueError("movie: rotate_pause_s must be >= 0, rotate_fov_deg between 0 and 180, "
                             "rotate_tilt_deg between -90 and 90 and rotate_pull_back > 0")
        # The camera must stay outside the block's circumscribed sphere (so every corner is in front of
        # it): face-on it is the block's half width / tan(fov / 2) + its half depth from the centre.
        if (1 / math.tan(math.radians(fov) / 2) + 1) * min(k or 1, 1) <= math.hypot(1, h / w, 1):
            raise ValueError(f"movie: with rotate_fov_deg {fov} and rotate_pull_back {k} the camera could cut "
                             "into the block: use a narrower field of view or pull back more")
    nz, ny, nx = shape0
    tx, ty, tz = (float(v) for v in m["target"])
    n = int(round(movie_seconds(m) * fps))
    t = np.arange(n) / fps
    zs = float(m["zoom_start"] or max(nx / w, ny / h))
    ze = float(m["zoom_end"])
    zo = float(m["zoom_out_to"] or max((nz if side else nx) / w, ny / h))
    ease = _ease(t, t0, t1)
    ease_out = _ease(t, t4, t5) if m["zoom_out_s"] else np.zeros(n)
    zoom_in = np.exp(np.log(zs) + (np.log(ze) - np.log(zs)) * ease)
    zoom_out = np.exp(np.log(ze) + (np.log(zo) - np.log(ze)) * ease_out)
    zoom = zoom_in * zoom_out / ze
    # Centre: the volume's centre at zoom_start, the target at zoom_end, with the target's offset from
    # the centre proportional to (zoom - zoom_end): it moves straight to the frame centre on screen.
    k = (zoom_in - ze) / (zs - ze) if zs != ze else np.zeros(n)
    cx = tx + (nx / 2 - tx) * k
    cy = ty + (ny / 2 - ty) * k
    out_speed, in_speed = speeds(m)
    speed = out_speed + (in_speed - out_speed) * (ease - ease_out)   # slices/s, with the zoom
    turn = _ease(t, t2, t3) if m["rotate_deg"] else np.zeros(n)
    if during:
        zr = float(m["rotate_start_zoom"])
        if not (side and m["zoom_out_s"] and not block and ze < zr < zo):
            raise ValueError(f"movie: rotate_start_zoom needs rotate_style plane, rotate_deg 90 or -90 and a "
                             f"zoom-out, and a zoom between {ze:g} and {zo:g}")
        ta = t4 + math.acos(1 - 2 * math.log(zr / ze) / math.log(zo / ze)) / math.pi * m["zoom_out_s"]
        if ta + m["rotate_s"] > t5 + 1e-9:
            raise ValueError(f"movie: the turn would end {ta + m['rotate_s'] - t5:.2f} s after the zoom-out: "
                             "shorten rotate_s or start it at a smaller rotate_start_zoom")
        turn = _ease(t, ta, ta + m["rotate_s"])
    theta = np.radians(m["rotate_deg"]) * turn
    if block:
        p = m["rotate_pause_s"]
        speed = speed * (1 - _ease(t, t2 - p, t2) + _ease(t, t3, t3 + p))
    # The view moves along the cutting plane's normal, (-sin, 0, cos) in (x, y, z), and passes the
    # target slice when the zoom ends (t1).
    d = m["direction"]
    cx = cx + _integral(-d * speed * np.sin(theta), t, t1)
    z = tz + _integral(d * speed * np.cos(theta), t, t1)
    half = np.zeros(n)
    if block:
        # Turned, the view faces the block's side: the y-z plane half a frame width to the side
        # (+x for +90), centred half the block's depth (= the frame's width) behind its front face.
        half = np.where(turn == 1, w / 2 * ze, 0.0)
        cx = cx + np.sign(m["rotate_deg"]) * half
        z = z + half
    if m["zoom_out_s"]:
        # The zoom-out mirrors the zoom-in: from the view at t4 straight to the final plane's centre. The
        # side view's z is where the scroll leaves it once turned: at t4 (interpolated without a block
        # turn's jump, which can fall between t4's frames), or at t5 for a turn during the zoom-out.
        ko = (zoom_out - ze) / (zo - ze) if zo != ze else np.zeros(n)
        cy = cy + (ny / 2 - np.interp(t4, t, cy)) * ko
        if side:
            z4 = np.interp(t5 if during else t4, t, z - half) + (w / 2 * ze if block else 0)
            z = z + ((nz - 1) / 2 - z4) * ko
        else:
            cx = cx + (nx / 2 - np.interp(t4, t, cx)) * ko
    # Up to the zoom-out every frame must lie in the volume (a turned frame spans w/2 * zoom in z).
    upto = t <= t4 + 1e-9
    span = np.where(upto, w / 2 * zoom * np.abs(np.sin(theta)), 0.0)
    lo, hi = z - span, z + span
    if block:                                    # the turning block reaches a frame width behind its front
        turning = (turn > 0) & (turn < 1)
        lo, hi = np.where(turning, z, lo), np.where(turning, z + w * zoom, hi)
    if lo.min() < -1e-6 or hi.max() > nz - 1 + 1e-6:
        lo, hi = (lo - tz).min(), (hi - tz).max()
        raise ValueError(f"movie: the frames reach from {lo:+.0f} to {hi:+.0f} slices around the target z, "
                         f"outside the volume's 0-{nz - 1}: target z must be between "
                         f"{math.ceil(-lo - 1e-6)} and {math.floor(nz - 1 - hi + 1e-6)} "
                         f"(or shorten the movie, or scroll slower)")
    return pd.DataFrame({"frame": np.arange(n), "t": t, "zoom": zoom, "cx": cx, "cy": cy, "z": z,
                         "rotation": np.degrees(theta), "turn": turn})


def speeds(m):
    """(zoomed-out, zoomed-in) scroll speed in slices/s."""
    s = m["slices_per_s"]
    return (float(s[0]), float(s[1])) if isinstance(s, (list, tuple)) else (float(s), float(s))


def movie_seconds(m):
    rotation = m["rotate_s"] + m["rotated_s"] if m["rotate_deg"] and m.get("rotate_start_zoom") is None else 0
    out = m["zoom_out_s"] + m["hold_out_s"] if m["zoom_out_s"] else 0
    return m["hold_start_s"] + m["zoom_s"] + m["hold_end_s"] + rotation + out


def open_levels(root, cache_gb):
    """Every scale of an OME-Zarr volume (s0, s1, ...), read-only, sharing one chunk cache."""
    context = ts.Context({"cache_pool": {"total_bytes_limit": int(cache_gb * 2 ** 30)}})
    levels = []
    while (Path(root) / f"s{len(levels)}" / "zarr.json").exists():
        levels.append(ts.open({"driver": "zarr3", "recheck_cached_data": "open",
                               "kvstore": {"driver": "file", "path": str(Path(root) / f"s{len(levels)}")}},
                              open=True, read=True, context=context).result())
    if not levels:
        raise FileNotFoundError(f"no OME-Zarr scales in {root}")
    return levels


def _read(arr, z0, z1, y0, y1, x0, x1):
    """arr[z0:z1, y0:y1, x0:x1] as float32, zeros where the box is outside the array."""
    out = np.zeros((z1 - z0, y1 - y0, x1 - x0), np.float32)
    nz, ny, nx = arr.shape
    za, zb, ya, yb, xa, xb = max(z0, 0), min(z1, nz), max(y0, 0), min(y1, ny), max(x0, 0), min(x1, nx)
    if za < zb and ya < yb and xa < xb:
        out[za - z0:zb - z0, ya - y0:yb - y0, xa - x0:xb - x0] = arr[za:zb, ya:yb, xa:xb].read().result()
    return out


def _level(levels, zoom):
    """The finest pyramid level that is not finer than ``zoom`` s0 voxels per frame px."""
    return int(np.clip(math.floor(math.log2(zoom) + 1e-9), 0, len(levels) - 1))


def render_frame(levels, size, zoom, cx, cy, z, rotation=0.0):
    """(uint8 frame, level used): the view of slice ``z`` centred on (cx, cy) at ``zoom`` s0 voxels
    per frame px, resampled from the finest level that is not finer than the frame needs; with a
    ``rotation`` (degrees), on the plane through that point turned about the y axis instead (at
    +-90: the y-z plane at x = cx, frame columns along +-z)."""
    if abs(rotation) < 1e-6:
        return _plane_frame(levels, size, zoom, cx, cy, z + 0.5, "z")
    if abs(abs(rotation) - 90) < 1e-6:
        frame, level = _plane_frame(levels, size, zoom, z + 0.5, cy, cx, "x")
        return (frame if rotation > 0 else np.ascontiguousarray(frame[:, ::-1])), level
    return _turned(levels, size, zoom, cx, cy, z, math.radians(rotation))


def _plane_frame(levels, size, zoom, uc, vc, cut, axis):
    """Frame on an axis-aligned plane. axis "z": the plane z = cut, frame columns along x (centre
    uc), rows along y (centre vc); axis "x": the plane x = cut, columns along z, rows along y;
    axis "y": the plane y = cut, columns along x, rows along z.
    Coordinates are continuous s0 voxels (voxel i spans [i, i+1)); between two planes of the level
    the two are blended, and downsampling is anti-aliased."""
    w, h = size
    level = _level(levels, zoom)
    arr, f = levels[level], 2 ** level
    r = zoom / f                                      # level px per frame px
    # Continuous s0 coordinate X -> level pixel index: X / f - 0.5.
    u_first = (uc + (0.5 - w / 2) * zoom) / f - 0.5   # frame px 0's centre
    v_first = (vc + (0.5 - h / 2) * zoom) / f - 0.5
    sigma = 0.5 * math.sqrt(r * r - 1) if r > 1.05 else 0.0   # anti-aliasing before downsampling
    pad = 2 + math.ceil(3 * sigma)
    u0, v0 = math.floor(u_first) - pad, math.floor(v_first) - pad
    u1, v1 = math.ceil(u_first + (w - 1) * r) + pad + 1, math.ceil(v_first + (h - 1) * r) + pad + 1
    n_cut = arr.shape["zyx".index(axis)]
    cl = min(max(cut / f - 0.5, 0.0), n_cut - 1.0)    # the cut on this level (between two planes)
    ca = int(math.floor(cl))
    frac = cl - ca
    two = frac > 1e-3 and ca + 1 < n_cut
    if axis == "z":
        planes = _read(arr, ca, ca + 1 + two, v0, v1, u0, u1)
    elif axis == "y":
        planes = _read(arr, v0, v1, ca, ca + 1 + two, u0, u1).transpose(1, 0, 2)   # (cut, z, x)
    else:
        planes = _read(arr, u0, u1, v0, v1, ca, ca + 1 + two).transpose(2, 1, 0)   # (cut, y, z)
    img = planes[0] * (1 - frac) + planes[1] * frac if two else np.ascontiguousarray(planes[0])
    if sigma:
        img = cv2.GaussianBlur(img, (0, 0), sigma)
    m = np.array([[r, 0, u_first - u0], [0, r, v_first - v0]], np.float64)
    interp = cv2.INTER_LINEAR if r >= 1 else cv2.INTER_CUBIC
    out = cv2.warpAffine(img, m, (w, h), flags=interp | cv2.WARP_INVERSE_MAP,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8), level


# Public names for the other movie modules (the private ones stay).
plane_frame, pyramid_level, ease, integral = _plane_frame, _level, _ease, _integral

STRIP = 32    # frame columns per read in _turned


def _turned(levels, size, zoom, cx, cy, z, theta):
    """A frame on the plane through (cx, cy, z) that contains the y axis and is turned by ``theta``
    (radians) from the x-y plane: frame columns run along (cos, 0, sin) in (x, y, z). Trilinear
    sampling of the level blurred along the plane as _plane_frame blurs it (anti-aliasing): along y,
    and along x and z as much as the plane runs along them, so at 0 and 90 degrees it matches."""
    w, h = size
    level = _level(levels, zoom)
    arr, f = levels[level], 2 ** level
    r = zoom / f
    sigma = 0.5 * math.sqrt(r * r - 1) if r > 1.05 else 0.0
    pad = math.ceil(4 * sigma)
    s = (np.arange(w) + 0.5 - w / 2) * zoom                       # s0 voxels from the centre
    xs = (cx + s * math.cos(theta)) / f - 0.5                     # level pixel indices
    zs = (z + 0.5 + s * math.sin(theta)) / f - 0.5
    ys = (cy + (np.arange(h) + 0.5 - h / 2) * zoom) / f - 0.5
    ya, yb = math.floor(ys[0]) - pad, math.floor(ys[-1]) + 2 + pad
    iy = (np.floor(ys).astype(int) - ya)[None, :]
    wy = (ys - np.floor(ys)).astype(np.float32)[None, :]
    out = np.empty((h, w), np.float32)
    for c0 in range(0, w, STRIP):
        x, zz = xs[c0:c0 + STRIP], zs[c0:c0 + STRIP]
        xa, za = math.floor(x.min()) - pad, math.floor(zz.min()) - pad
        box = _read(arr, za, math.floor(zz.max()) + 2 + pad, ya, yb, xa, math.floor(x.max()) + 2 + pad)
        if sigma:
            box = ndimage.gaussian_filter(box, (sigma * abs(math.sin(theta)), sigma, sigma * abs(math.cos(theta))),
                                          mode="nearest", truncate=4.0)
        ix, iz = np.floor(x).astype(int) - xa, np.floor(zz).astype(int) - za
        wx, wz = (x - np.floor(x)).astype(np.float32), (zz - np.floor(zz)).astype(np.float32)
        acc = np.zeros((len(x), h), np.float32)
        for dz, gz in ((0, 1 - wz), (1, wz)):
            for dx, gx in ((0, 1 - wx), (1, wx)):
                bz, bx = (iz + dz)[:, None], (ix + dx)[:, None]
                line = box[bz, iy, bx] * (1 - wy) + box[bz, iy + 1, bx] * wy      # (columns, rows)
                acc += (gz * gx)[:, None] * line
        out[:, c0:c0 + STRIP] = acc.T
    return np.clip(np.rint(out), 0, 255).astype(np.uint8), level


def camera(centre, yaw, tilt, distance, up):
    """(position, forward, right, down) of a camera ``distance`` from ``centre`` (s0 voxels) and
    looking at it: along +z (frame columns along +x, rows along +y) turned by ``yaw`` (radians) about
    the y axis, as the frames turn (columns along (cos, 0, sin)), and raised by ``tilt`` towards
    ``up`` (-1: -y, 1: +y, the direction that is up on screen)."""
    e = np.array([0.0, up, 0.0])
    forward = math.cos(tilt) * np.array([-math.sin(yaw), 0.0, math.cos(yaw)]) - math.sin(tilt) * e
    down = np.array([0.0, 1.0, 0.0]) - forward[1] * forward
    down /= np.linalg.norm(down)
    return np.asarray(centre, float) - distance * forward, forward, np.cross(down, forward), down


def project(points, cam, size, focal):
    """Frame coordinates (continuous px, (0, 0) the top-left corner) of s0 voxel ``points``."""
    pos, forward, right, down = cam
    q = np.asarray(points, float) - pos
    depth = q @ forward
    return np.stack([size[0] / 2 + focal * (q @ right) / depth, size[1] / 2 + focal * (q @ down) / depth], -1)


COVER = 4       # render_faces: coverage samples per frame px along each axis
MAX_SHRINK = 64  # render_faces: anti-aliasing blurs for shrinks up to this


def render_faces(faces, size, cam, focal, shade=0.0, coverage=False):
    """Perspective picture, on black, of flat textured rectangles: ``faces`` [(image, corner, du, dv,
    normal)], the image's top-left corner at ``corner`` (s0 voxels), its width spanning ``du`` and its
    height ``dv``. Faces turned away from the camera are left out (so a convex block needs no depth
    order) and later faces cover earlier ones; a face seen obliquely is darkened by up to ``shade``
    (0: the data's grey values as they are). With ``coverage``: (picture, the fraction of each frame
    px the faces cover).
    Anti-aliased: each image is blurred, pixel by pixel, as much as the view shrinks it there, and
    each frame px shows its faces in proportion to COVER x COVER coverage samples (so faces sharing
    an edge leave no seam)."""
    w, h = size
    owner = np.zeros((h * COVER, w * COVER), np.uint16)   # the face each coverage sample shows (0: none)
    drawn = []
    for img, corner, du, dv, normal in faces:
        corner, du, dv = (np.asarray(v, float) for v in (corner, du, dv))
        to_cam = cam[0] - (corner + du / 2 + dv / 2)
        facing = float(np.dot(normal, to_cam)) / np.linalg.norm(to_cam)
        if facing <= 0:
            continue
        dst = project([corner, corner + du, corner + du + dv, corner + dv], cam, size, focal)
        rows, cols = img.shape
        src = np.float32([[-0.5, -0.5], [cols - 0.5, -0.5], [cols - 0.5, rows - 0.5], [-0.5, rows - 0.5]])
        m = cv2.getPerspectiveTransform(src, (dst - 0.5).astype(np.float32))
        # Lanczos (bilinear softens the image at once when the view starts to move off 1:1), held within
        # the range of the 3 x 3 image px around each sample: no ringing, no grey value the data lack there
        img = _antialiased(img.astype(np.float32), m)
        tex, lo, hi = (cv2.warpPerspective(a, m, (w, h), flags=f, borderMode=cv2.BORDER_REPLICATE) for a, f in
                       ((img, cv2.INTER_LANCZOS4), (cv2.erode(img, np.ones((3, 3))), cv2.INTER_NEAREST),
                        (cv2.dilate(img, np.ones((3, 3))), cv2.INTER_NEAREST)))
        tex = np.clip(tex, lo, hi)
        box = _fill(owner, dst * COVER - 0.5, len(drawn) + 1)
        if box:
            drawn.append((tex * (1 - shade * (1 - facing)), box))
    out, covered = np.zeros((h, w), np.float32), np.zeros((h, w), np.float32)
    for i, (tex, (r0, r1, c0, c1)) in enumerate(drawn, 1):
        r0, r1, c0, c1 = r0 // COVER, -(-r1 // COVER), c0 // COVER, -(-c1 // COVER)     # whole frame px
        mine = owner[r0 * COVER:r1 * COVER, c0 * COVER:c1 * COVER] == i
        cover = cv2.resize(mine.astype(np.float32), (c1 - c0, r1 - r0), interpolation=cv2.INTER_AREA)
        out[r0:r1, c0:c1] += cover * tex[r0:r1, c0:c1]
        covered[r0:r1, c0:c1] += cover
    return (out, covered) if coverage else out


def _fill(owner, corners, value):
    """Set the samples of ``owner`` whose centres lie in the convex polygon ``corners`` (sample (i, j)
    centred on (j, i)) to ``value``; returns the rows and columns set (start, stop, start, stop), or
    None. Each row is filled from ceil(left edge) to below ceil(right edge), so polygons sharing an edge
    neither overlap nor leave a gap (cv2.fillPoly includes both ends)."""
    rows, cols = owner.shape
    y = np.arange(rows, dtype=float)
    lo, hi = np.full(rows, -np.inf), np.full(rows, np.inf)
    centre = corners.mean(axis=0)
    for (xa, ya), (xb, yb) in zip(corners, np.roll(corners, -1, axis=0)):
        a, b = yb - ya, xa - xb                      # inside: a x + b y + c >= 0
        c = -(a * xa + b * ya)
        if a * centre[0] + b * centre[1] + c < 0:
            a, b, c = -a, -b, -c
        if a > 0:
            lo = np.maximum(lo, -(b * y + c) / a)
        elif a < 0:
            hi = np.minimum(hi, -(b * y + c) / a)
        else:
            hi = np.where(b * y + c >= 0, hi, -np.inf)
    j0, j1 = (np.clip(np.ceil(v), 0, cols).astype(int) for v in (lo, hi))
    filled = np.flatnonzero(j1 > j0)
    for i in filled:
        owner[i, j0[i]:j1[i]] = value
    return (filled[0], filled[-1] + 1, j0[filled].min(), j1[filled].max()) if len(filled) else None


def _antialiased(img, m):
    """``img`` blurred, pixel by pixel and along each axis, as much as the homography ``m`` (image px
    -> frame px) shrinks it there (sigma as in _plane_frame): blends of the two nearest of a ladder of
    blurs, shrink 2^(k/2) apart."""
    rows, cols = img.shape
    y, x = np.mgrid[0:rows + 15:16, 0:cols + 15:16].astype(float)       # a coarse grid; shrink is smooth
    X, Y, Z = (m[i, 0] * x + m[i, 1] * y + m[i, 2] for i in range(3))
    with np.errstate(divide="ignore", invalid="ignore"):   # an edge-on face: det (and Z) 0 -> inf, 0/0 -> nan
        a, b = (m[0, 0] * Z - X * m[2, 0]) / Z ** 2, (m[0, 1] * Z - X * m[2, 1]) / Z ** 2   # frame px per image px
        c, d = (m[1, 0] * Z - Y * m[2, 0]) / Z ** 2, (m[1, 1] * Z - Y * m[2, 1]) / Z ** 2
        det = np.abs(a * d - b * c)
        shrinks = np.hypot(b, d) / det, np.hypot(a, c) / det                     # image px per frame px
    for axis, shrink in zip((1, 0), shrinks):
        level = 2 * np.log2(np.clip(np.nan_to_num(shrink, nan=MAX_SHRINK), 1, MAX_SHRINK)).astype(np.float32)
        level = cv2.resize(level, (cols, rows), interpolation=cv2.INTER_LINEAR)
        if level.max() < 0.14:                                      # shrink < 1.05 everywhere
            continue
        out = np.zeros_like(img)
        for k in range(math.ceil(level.max()) + 1):
            weight = np.clip(1 - np.abs(level - k), 0, 1)
            if k == 0:
                out += weight * img
            elif weight.any():
                sigma = 0.5 * math.sqrt(2 ** k - 1)
                n = 2 * math.ceil(3 * sigma) + 1
                out += weight * cv2.GaussianBlur(img, (n, 1) if axis else (1, n), sigma, sigma)
        img = out
    return img


def block_faces(levels, size, zoom, cx, cy, z, side, up, top=True):
    """The block a turn shows: the frame (slice ``z`` around (cx, cy) at ``zoom``) as its front face,
    as deep along +z as the frame is wide. Its faces for ``render_faces``: the front, the y-z plane on
    ``side`` (1: +x, -1: -x; exactly the frame a turn to +-90 degrees ends on) and, if ``top``, the x-z
    plane on the ``up`` edge (-1: -y; the side the camera rises to, up on screen for a positive tilt)."""
    w, h = size
    hw, hh = w / 2 * zoom, h / 2 * zoom
    zf = z + 0.5                                      # the slice's plane (continuous s0 voxels)
    xs = cx + side * hw
    front = render_frame(levels, size, zoom, cx, cy, z)[0]
    flank = render_frame(levels, size, zoom, xs, cy, z + hw, 90 * side)[0]
    faces = [(front, (cx - hw, cy - hh, zf), (2 * hw, 0, 0), (0, 2 * hh, 0), (0, 0, -1)),
             (flank, (xs, cy - hh, zf + hw - side * hw), (0, 0, 2 * side * hw), (0, 2 * hh, 0), (side, 0, 0))]
    if top:
        yt = cy + up * hh
        img = _plane_frame(levels, (w, w), zoom, cx, zf + hw, yt, "y")[0]
        faces.append((img, (cx - hw, yt, zf), (2 * hw, 0, 0), (0, 0, 2 * hw), (0, up, 0)))
    return {"faces": faces, "centre": (cx, cy, zf + hw), "half": (hw, hh, hw)}


def block_view(block, size, turn, rotate_deg, fov_deg, tilt_deg, pull_back, up):
    """A frame of the block (``block_faces``) turning: ``turn`` 0-1 through the turn by ``rotate_deg``.
    The camera orbits the block's centre; at 0 the front face fills the frame and at 1 the side face,
    exactly; in between it pulls back by up to ``pull_back`` (null: the whole block fits the frame at
    mid-turn, with a margin) and rises by up to ``tilt_deg``, both easing in and out."""
    w, h = size
    tan = math.tan(math.radians(fov_deg) / 2)
    hw = block["half"][0]
    focal, dist0 = w / 2 / tan, hw / tan + hw         # face-on: the near face fills the frame
    lift = math.sin(math.pi * turn)

    def cam(k, u, lift):
        return camera(block["centre"], math.radians(rotate_deg) * u, math.radians(tilt_deg) * lift,
                      dist0 * (1 + (k - 1) * lift), up)

    if pull_back is None:
        if "pull_back" not in block:
            corners = np.array(block["centre"]) + np.array(
                [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * block["half"]
            margin = 0.06 * h
            lo, hi = 1.0, 8.0
            for _ in range(40):
                k = (lo + hi) / 2
                p = project(corners, cam(k, 0.5, 1.0), size, focal)
                inside = (p.min(0) >= margin).all() and (p[:, 0].max() <= w - margin) and (p[:, 1].max() <= h - margin)
                lo, hi = (lo, k) if inside else (k, hi)
            block["pull_back"] = hi
        pull_back = block["pull_back"]
    out = render_faces(block["faces"], size, cam(pull_back, turn, lift), focal, shade=0.0)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def nice_length(target):
    """Largest 1, 2 or 5 x 10^k not above ``target`` (> 0)."""
    p = 10 ** math.floor(math.log10(target))
    return max(v * p for v in (1, 2, 5) if v * p <= target * (1 + 1e-9))


def draw_scale_bar(frame, nm_per_px):
    """Scale bar (white, outlined) in the bottom-right corner, about a tenth of the frame wide."""
    from PIL import Image, ImageDraw, ImageFont
    import matplotlib

    h, w = frame.shape
    length_nm = nice_length(w / 10 * nm_per_px)
    length_px = length_nm / nm_per_px
    label = f"{length_nm / 1000:g} µm" if length_nm >= 1000 else f"{length_nm:g} nm"
    margin, thick = round(h / 18), max(4, round(h / 150))
    x1, y1 = w - margin, h - margin
    x0, y0 = x1 - length_px, y1 - thick
    font = ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"), round(h / 28))
    im = Image.fromarray(frame)
    d = ImageDraw.Draw(im)
    d.rectangle([x0 - 2, y0 - 2, x1 + 2, y1 + 2], fill=0)
    d.rectangle([x0, y0, x1, y1], fill=255)
    d.text(((x0 + x1) / 2, y0 - thick), label, fill=255, font=font, anchor="ms", stroke_width=3, stroke_fill=0)
    return np.asarray(im)


def frames_dir(cfg):
    return step_dir(cfg, "movie", movie_name(cfg))


def make_frames(cfg, only=None, overwrite=False):
    """Write the PNG frames (all, or the frame numbers in ``only``); frames from an earlier run with
    the same timeline are kept, a changed timeline starts afresh."""
    m = cfg["movie"]
    root = volume_path(cfg)
    levels = open_levels(root, m["cache_gb"])
    plan = timeline(m, levels[0].shape)
    out = frames_dir(cfg)
    plan_csv = out / "timeline.csv"
    keys = ("size", "scale_bar", "view_rotation_deg", "rotate_fov_deg", "rotate_tilt_deg", "rotate_pull_back")
    settings = {k: m[k] for k in keys} | {"volume": str(root)}
    plan_text = f"# {settings}\n" + plan.to_csv(index=False, float_format="%.6f")
    if overwrite or not plan_csv.exists() or plan_csv.read_text() != plan_text:
        for old in out.glob("f*.png"):
            old.unlink()
        cli.atomic_write(plan_csv, lambda p: Path(p).write_text(plan_text))
    voxel = serve.voxel_nm(root) or {"x": 8.0}
    rows = plan if only is None else plan[plan["frame"].isin(only)]
    log.info("%s: %d frames of %d (%dx%d, %.1f s at %d fps), zoom %.1f -> %.2f voxels/px, z %.0f -> %.0f",
             root.name, len(rows), len(plan), *m["size"], movie_seconds(m), m["fps"],
             plan["zoom"].iloc[0], plan["zoom"].iloc[-1], plan["z"].iloc[0], plan["z"].iloc[-1])
    up = -1 if m["view_rotation_deg"] == 0 else 1     # the y direction that is up on screen
    rises = up if m["rotate_tilt_deg"] >= 0 else -up  # the side the turning camera rises to
    blocks = {}
    done = 0
    for row in rows.itertuples():
        path = out / f"f{row.frame:05d}.png"
        if path.exists():
            continue
        if m["rotate_style"] == "block" and 0 < row.turn < 1:
            key = (row.zoom, row.cx, row.cy, row.z)
            if key not in blocks:
                blocks = {key: block_faces(levels, m["size"], *key, np.sign(m["rotate_deg"]), rises,
                                           top=m["rotate_tilt_deg"] != 0)}
            frame = block_view(blocks[key], m["size"], row.turn, m["rotate_deg"], m["rotate_fov_deg"],
                               m["rotate_tilt_deg"], m["rotate_pull_back"], up)
            level = _level(levels, row.zoom)
        else:
            frame, level = render_frame(levels, m["size"], row.zoom, row.cx, row.cy, row.z, row.rotation)
        if m["view_rotation_deg"] == 180:
            frame = np.ascontiguousarray(np.rot90(frame, 2))     # about the frame centre, where the target is
        if m["scale_bar"]:
            frame = draw_scale_bar(frame, row.zoom * voxel["x"])
        cli.atomic_write(path, lambda p: cv2.imwrite(p, frame, [cv2.IMWRITE_PNG_COMPRESSION, 1]))
        done += 1
        if done % 50 == 0:
            log.info("frame %d/%d (zoom %.2f from s%d, z %.1f, turned %.0f deg)", row.frame + 1, len(plan),
                     row.zoom, level, row.z, row.rotation)
    log.info("%d frames written to %s", done, out)
    return out


def encode_command(cfg):
    m = cfg["movie"]
    out = Path(cfg["output_dir"]) / f"{movie_name(cfg)}.mp4"
    return [m["ffmpeg"], "-y", "-loglevel", "warning", "-framerate", str(m["fps"]),
            "-i", str(frames_dir(cfg) / "f%05d.png"), "-c:v", "libx264", "-preset", "slow", "-crf", str(m["crf"]),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]


def encode(cfg):
    m = cfg["movie"]
    n = int(round(movie_seconds(m) * m["fps"]))
    missing = [k for k in range(n) if not (frames_dir(cfg) / f"f{k:05d}.png").exists()]
    if missing:
        raise RuntimeError(f"{len(missing)} of {n} frames missing (first: {missing[0]}): run 'frames' first")
    cmd = encode_command(cfg)
    subprocess.run(cmd, check=True)
    log.info("wrote %s", cmd[-1])


def main(argv=None):
    p = cli.base_parser(__doc__.split("\n\n")[0])
    p.add_argument("action", choices=("frames", "encode"))
    p.add_argument("--frames", help="frames: only these frame numbers, e.g. 0,240,480 (a quick look)")
    p.add_argument("--print", action="store_true", help="encode: print the ffmpeg command instead of running it")
    args = p.parse_args(argv)
    cfg = cli.setup(args, DEFAULTS)
    if args.action == "frames":
        only = [int(v) for v in args.frames.split(",")] if args.frames else None
        make_frames(cfg, only, args.overwrite)
    elif args.print:
        print(shlex.join(encode_command(cfg)))
    else:
        if not shutil.which(cfg["movie"]["ffmpeg"]):
            raise FileNotFoundError(f"{cfg['movie']['ffmpeg']} not found (on NEMO: ml FFmpeg)")
        encode(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
