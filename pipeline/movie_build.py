"""Acquisition movie: the volume built up slice by slice in 3D, in acquisition order, then the
fly-through (pipeline.movie).

The stack starts with the fly-through's first slice and grows by whole slices towards +z, the order
they were imaged in (check numbers slices by time, render keeps that order: z = 0 is the first). The
camera starts close to a corner of the newest slice, from the +z side, where a single 8 nm slice is a
pixel thick, and pulls back as the imaging speeds up to thousands of slices a second, until the whole
block is in view. When the last slice is in, the camera orbits round the block's side to its other
end, the first slice, and comes to rest face-on to it, exactly the fly-through's first frame; the
fly-through (the ``movie:`` settings: zoom into the target, scrolling through z) follows without a cut.

Everything on screen is the volume, resampled: the block's faces are the volume's planes on them,
drawn in perspective by pipeline.movie.render_faces (no shading), each at the resolution the view
needs (``Stack``). The block is the tissue's footprint (one box per tile layout, ``footprints``), cut
just inside it so its sides show data rather than the empty margin; the slice's rim outside the block
fades in during the last second before the fly-through, which shows the whole slice.

    python -m pipeline.movie_build --config C frames     # PNG frames -> work/movie/<name>/
    python -m pipeline.movie_build --config C encode     # ffmpeg -> output_dir/<name>.mp4
    sbatch slurm/movie_build.sbatch C                    # both, on the cluster

Coordinates are continuous s0 voxels (x, y, z) as in pipeline.movie; slice z fills z to z + 1.
"""

import logging
import math
import shlex
import shutil
import subprocess
import sys
from collections import OrderedDict, namedtuple
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from . import cli, movie, serve
from .config import deep_merge

log = logging.getLogger(__name__)

DEFAULTS = {
    "build": {
        "z_far": None,            # the stack's last slice; null: as far as the tissue keeps the footprint it has
                                  # on the fly-through's first slice (a smaller one later makes a step)
        "tissue_fill": 0.98,      # each side of the block is at least this much tissue (non-zero data)
        # Timeline (s)
        "lead_in_s": 0.3,         # black before the first slice
        "start_rate": 4.0,        # slices/s at first ...
        "slow_s": 1.5,            # ... for this long, then the rate rises (steadily in log scale, eased in and
        "ramp_s": 6.0,            # out) over this long to max_rate, less if the stack is too thin for it
        "max_rate": 3000,
        "stop_s": 1.2,            # slowing to a stop on the last slice
        "orbit_delay_s": 0.4,     # the camera orbits to the first slice from this long after the slowing starts
        "orbit_s": 4.5,           # ... over this long (it must end after the imaging)
        "fade_s": 1.2,            # the slice's rim outside the block fades in over the orbit's last this long
        "scroll_ease_s": 1.0,     # the fly-through's scroll speeds up from rest over this long (0: it starts at
                                  # full speed); <= movie.hold_start_s
        # Camera: yaw as pipeline.movie.camera (0: looking along +z, at the first slice; 180: at the newest);
        # the orbit runs on to 360 (via 270: from -x) or back to 0 (via 90), whichever is nearer
        "close_up": 1.5,          # voxels per frame px on the newest slice at the start, near the block's corner
        "yaw_deg": [215, 225],    # at the start and once the block is whole
        "tilt_deg": [25, 22],     # elevation, towards up on screen (the top of the block shows)
        "fov_deg": 50,            # horizontal field of view
        "fill": 0.84,             # the whole block's share of the frame height (or width)
    },
}

Box = namedtuple("Box", "za zb x0 x1 y0 y1")    # slices za..zb-1 (z from za to zb), x0 <= x < x1, y0 <= y < y1
FACE_AXES = {0: (2, 1, "x"), 1: (0, 2, "y"), 2: (0, 1, "z")}   # normal axis: (texture column axis, row axis,
                                                               # movie._plane_frame axis)


def footprints(levels, z0, z1, fill=0.98, tol=0.02, min_depth=0.015):
    """The tissue's footprint over slices z0..z1-1: Boxes (s0 voxels) whose four sides are each at least
    ``fill`` tissue (non-zero) in every slice, one per stretch of slices where the footprint stays the
    same to ``tol`` of the slice's size (stretches under ``min_depth`` of the volume's depth join the
    one before). Read from the coarsest level still >= 512 px wide (median over 9 of its slices),
    edges one voxel of it in."""
    level = max([0] + [k for k, a in enumerate(levels) if a.shape[2] >= 512])
    arr, f = levels[level], 2 ** level
    nz, ny, nx = levels[0].shape
    tol, min_slices = tol * max(nx, ny), min_depth * nz
    pa, pb = z0 // f, min(-(-z1 // f), arr.shape[0])
    boxes = np.array([_tissue_box(plane, fill) for plane in arr[pa:pb].read().result() > 0], float)
    if np.isnan(boxes).any():
        raise ValueError(f"build: no tissue in some slices between {z0} and {z1}")
    k = min(9, len(boxes)) | 1                                   # (planes, 4): c0 c1 r0 r1
    pad = np.pad(boxes, ((k // 2, k // 2), (0, 0)), mode="edge")
    boxes = np.median(np.lib.stride_tricks.sliding_window_view(pad, k, axis=0), axis=-1)
    starts = [0]
    for p in range(1, len(boxes)):
        if np.abs(boxes[p] - _meet(boxes[starts[-1]:p])).max() * f > tol:
            starts.append(p)
    starts = [p for i, p in enumerate(starts) if i == 0 or (p - starts[i - 1]) * f >= min_slices]
    ends = starts[1:] + [len(boxes)]
    for i in range(1, len(starts)):
        # The coarse slice where the footprint changes holds some of each, so it looks like the larger:
        # it goes with the smaller one.
        before, after = _meet(boxes[starts[i - 1]:starts[i]]), _meet(boxes[starts[i]:ends[i]])
        shift = -1 if np.prod(before[[1, 3]] - before[[0, 2]]) > np.prod(after[[1, 3]] - after[[0, 2]]) else 1
        if starts[i - 1] < starts[i] + shift < ends[i]:
            starts[i] += shift
            ends[i - 1] = starts[i]
    out = []
    for first, end in zip(starts, ends):
        c0, c1, r0, r1 = _meet(boxes[first:end])
        za, zb = max(z0, (pa + first) * f), min(z1, (pa + end) * f)
        out.append(Box(int(za), int(zb), *(float(v) * f for v in (c0 + 1, c1 - 1, r0 + 1, r1 - 1))))
    return out


def _meet(boxes):
    """The intersection of (c0, c1, r0, r1) boxes."""
    return np.r_[boxes[:, 0].max(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].min()]


def _tissue_box(mask, fill):
    """(c0, c1, r0, r1), half-open, of the rectangle in a 2D mask whose edges are each >= ``fill`` set:
    from the rows and columns more than half set, the emptiest edge moved in until all are full."""
    rows, cols = np.flatnonzero(mask.mean(1) > 0.5), np.flatnonzero(mask.mean(0) > 0.5)
    if not len(rows) or not len(cols):
        return (np.nan,) * 4
    r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    while r1 - r0 > 2 and c1 - c0 > 2:
        sub = mask[r0:r1, c0:c1]
        edges = [sub[:, 0].mean(), sub[:, -1].mean(), sub[0].mean(), sub[-1].mean()]
        worst = int(np.argmin(edges))
        if edges[worst] >= fill:
            break
        c0, c1, r0, r1 = c0 + (worst == 0), c1 - (worst == 1), r0 + (worst == 2), r1 - (worst == 3)
    return c0, c1, r0, r1


class Stack:
    """A stack of slices as solid boxes (``footprints``), each face showing the volume's plane on it
    (the voxels just inside), for pipeline.movie.render_faces: the stack can grow at either end and be
    seen from any side. Each view gets, per face, just the part in view at the resolution it needs:
    texels of zoom / 2^j s0 voxels (zoom: the coarsest; at least 1 voxel, at most ``max_texels`` a
    face), on a grid through ``origin`` (x, y), e.g. the pixel grid of a frame the stack must match,
    and through each box's first slice in z; box edges move inwards onto that grid. Where two boxes
    meet, the ends keep ``pad`` slices (the reach of the coarsest level read) away from the changeover,
    so neither shows the other's empty margin."""

    def __init__(self, levels, boxes, zoom, origin=(0.0, 0.0), max_texels=4e6):
        self.levels, self.zoom, self.origin, self.max_texels = levels, float(zoom), origin, max_texels
        self.finest = max(0, math.floor(math.log2(zoom)))      # the largest j
        self.pad = 3 * 2 ** movie._level(levels, zoom)

        def snap(lo, hi, g):
            return g + zoom * math.ceil((lo - g) / zoom - 1e-6), g + zoom * math.floor((hi - g) / zoom + 1e-6)

        self.boxes = []
        for b in boxes:
            (x0, x1), (y0, y1) = snap(b.x0, b.x1, origin[0]), snap(b.y0, b.y1, origin[1])
            if x1 - x0 < zoom or y1 - y0 < zoom:
                raise ValueError(f"build: footprint {b} is smaller than a texel ({zoom:g} voxels)")
            self.boxes.append(Box(float(b.za), float(b.zb), x0, x1, y0, y1))
        self._cache = OrderedDict()

    def corners(self, lo, hi):
        """Corners (n, 3) of the boxes between the planes z = ``lo`` and ``hi``."""
        out = []
        for b in self.boxes:
            a, c = max(lo, b.za), min(hi, b.zb)
            if c >= a:
                out += [(x, y, z) for x in (b.x0, b.x1) for y in (b.y0, b.y1) for z in (a, c)]
        return np.array(out, float)

    def faces(self, lo, hi, cam, size, focal):
        """Faces (image, corner, du, dv, normal) of the stack between the planes z = ``lo`` < ``hi`` (the
        slices lo..hi-1) as camera ``cam`` (pipeline.movie.camera) sees it on a frame of ``size`` with
        focal length ``focal`` px, far to near (right for any camera outside the boxes). The ends show
        the slices lo and hi - 1."""
        pos = np.asarray(cam[0], float)
        live = [i for i, b in enumerate(self.boxes) if max(lo, b.za) < min(hi, b.zb)]
        live.sort(key=lambda i: -max(self.boxes[i].za - pos[2], pos[2] - self.boxes[i].zb, 0.0))
        out = []
        for i in live:
            b = self.boxes[i]
            za, zb = max(lo, b.za), min(hi, b.zb)
            for axis, at, sign in ((0, b.x0, -1), (0, b.x1, 1), (1, b.y0, -1), (1, b.y1, 1), (2, za, -1), (2, zb, 1)):
                if (pos[axis] - at) * sign > 0:                  # else facing away
                    face = self._face(i, axis, at, sign, (za, zb), cam, size, focal)
                    if face is not None:
                        out.append(face)
        return out

    def _face(self, i, axis, at, sign, zs, cam, size, focal):
        """Box ``i``'s face on the plane ``axis`` = ``at`` (normal ``sign``; ``zs``: its z extent), the part
        in view (None if none)."""
        b = self.boxes[i]
        iu, iv, plane = FACE_AXES[axis]
        extent = {0: (b.x0, b.x1), 1: (b.y0, b.y1), 2: zs}
        grid = {0: self.origin[0], 1: self.origin[1], 2: b.za}

        def point(u, v):
            p = np.zeros(3)
            p[axis], p[iu], p[iv] = at, u, v
            return p

        (u0, u1), (v0, v1) = extent[iu], extent[iv]
        seen = _clip(np.array([point(u0, v0), point(u1, v0), point(u1, v1), point(u0, v1)]), cam, size, focal)
        if len(seen) < 3:
            return None
        need = ((seen - cam[0]) @ cam[1]).min() / focal         # voxels per frame px where the face is nearest
        j = int(np.clip(math.ceil(math.log2(self.zoom / need) - 0.25), 0, self.finest))
        while True:
            t = self.zoom / 2 ** j
            # What is in view, with a margin, outwards onto the texel grid (g), and that within the face (f):
            # the same except at a stack end off the grid in z, which _crop keeps exact.
            region = []
            for k in (iu, iv):
                s0 = max(seen[:, k].min() - 4 * t, extent[k][0])
                s1 = min(seen[:, k].max() + 4 * t, extent[k][1])
                g0 = grid[k] + t * math.floor((s0 - grid[k]) / t + 1e-6)
                g1 = grid[k] + t * math.ceil((s1 - grid[k]) / t - 1e-6)
                region.append((g0, g1, max(g0, extent[k][0]), min(g1, extent[k][1])))
            n = [max(1, round((r[1] - r[0]) / t)) for r in region]
            if n[0] * n[1] <= self.max_texels or j == 0:
                break
            j -= 1
        (gu0, gu1, fu0, fu1), (gv0, gv1, fv0, fv1) = region
        if axis < 2:
            cut = at - 0.5 * sign                               # the voxels just inside
        else:
            cut = self._slice(i, at if sign < 0 else at - 1)    # the end slice
        key = (i, axis, sign, j, gu0, gu1, gv0, gv1, cut)
        if key not in self._cache:
            if len(self._cache) > 16:
                self._cache.popitem(last=False)
            self._cache[key] = movie._plane_frame(self.levels, tuple(n), t, (gu0 + gu1) / 2, (gv0 + gv1) / 2,
                                                  cut, plane)[0].astype(np.float32)
        tex = self._cache[key]
        if iu == 2:
            tex = _crop(tex, 1, (fu0 - gu0) / t, (fu1 - fu0) / t)
        if iv == 2:
            tex = _crop(tex, 0, (fv0 - gv0) / t, (fv1 - fv0) / t)
        normal = np.zeros(3)
        normal[axis] = sign
        corner = point(fu0, fv0)
        quad = np.array([corner, point(fu1, fv0), point(fu1, fv1), point(fu0, fv1)])
        if ((quad - cam[0]) @ cam[1]).min() <= 0:
            raise ValueError("build: the camera is too close to a face (part of it would be behind the camera)")
        return tex, corner, point(fu1, fv0) - corner, point(fu0, fv1) - corner, normal

    def _slice(self, i, z):
        """The plane (z + 0.5) of slice ``z`` on an end of box ``i``, kept ``pad`` away from another box."""
        b = self.boxes[i]
        lo, hi = b.za + (self.pad if i > 0 else 0), b.zb - 1 - (self.pad if i + 1 < len(self.boxes) else 0)
        return (min(max(z, lo), hi) if lo <= hi else (lo + hi) / 2) + 0.5


def _clip(poly, cam, size, focal, near=1.0):
    """The part of the convex polygon ``poly`` (n, 3) in view: in front of the camera and on the frame."""
    pos, forward, right, down = cam
    w, h = size
    for normal, offset in ((forward, near), (focal * right + w / 2 * forward, 0), (w / 2 * forward - focal * right, 0),
                           (focal * down + h / 2 * forward, 0), (h / 2 * forward - focal * down, 0)):
        if not len(poly):
            break
        d = (poly - pos) @ normal - offset
        out = []
        for k in range(len(poly)):
            nxt = (k + 1) % len(poly)
            if d[k] >= 0:
                out.append(poly[k])
            if (d[k] >= 0) != (d[nxt] >= 0):
                out.append(poly[k] + (poly[nxt] - poly[k]) * d[k] / (d[k] - d[nxt]))
        poly = np.array(out)
    return poly


def _crop(tex, axis, start, length):
    """Texels ``start`` to ``start + length`` (fractional) of ``tex`` along ``axis``, resampled to a whole
    number of texels (at least one); a plain slice when both ends are whole."""
    n = max(1, math.ceil(length - 1e-6))
    if abs(start - round(start)) < 1e-6 and abs(length - n) < 1e-6:
        a = round(start)
        return np.ascontiguousarray(tex[:, a:a + n] if axis == 1 else tex[a:a + n])
    s = length / n
    shift = start + 0.5 * s - 0.5
    if axis == 1:
        m, size = np.float64([[s, 0, shift], [0, 1, 0]]), (n, tex.shape[0])
    else:
        m, size = np.float64([[1, 0, 0], [0, s, shift]]), (tex.shape[1], n)
    return cv2.warpAffine(tex, m, size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)


def fly_plan(m, b, shape0):
    """The fly-through (pipeline.movie.timeline) the build-up hands over to, its scroll speeding up
    from rest over ``scroll_ease_s`` (its first slice moves along, the rest stays as it was)."""
    if m["rotate_deg"] and m["rotate_style"] == "block":
        raise ValueError("movie: a build movie ends with the fly-through without a block turn (rotate_deg 0)")
    plan = movie.timeline(m, shape0)
    e = b["scroll_ease_s"]
    if e:
        if e > m["hold_start_s"]:
            raise ValueError(f"build.scroll_ease_s ({e}) must be <= movie.hold_start_s ({m['hold_start_s']})")
        t = plan["t"].to_numpy()
        lag = m["direction"] * movie.speeds(m)[0] * (1 - movie._ease(t, 0, e))
        plan["z"] = plan["z"] - movie._integral(lag, t, e)
    return plan


def imaging(b, n_total):
    """Slices imaged against time, on a 1 ms grid: (t, n, rate, t_stop), n continuous (the stack holds
    floor(n) slices). Nothing for lead_in_s, then the first slice, start_rate for slow_s, the rate
    rising steadily in log scale (eased in and out) over ramp_s to max_rate (lower if the stack is too
    thin to get there), held, and from t_stop slowing to a stop over stop_s, exactly on n_total."""
    lead, stop, r0 = b["lead_in_s"], b["stop_s"], float(b["start_rate"])
    if min(lead, b["slow_s"], b["ramp_s"], stop) < 0 or r0 <= 0:
        raise ValueError("build: the times must be >= 0 and start_rate > 0")
    t1 = lead + b["slow_s"]
    t2 = t1 + b["ramp_s"]

    def run(rmax, hold):
        t3 = t2 + hold
        t = np.append(np.arange(0, t3 + stop, 1e-3), t3 + stop)
        r = np.exp(np.log(r0) + np.log(rmax / r0) * movie._ease(t, t1, t2)) * (1 - movie._ease(t, t3, t3 + stop))
        r[t < lead] = 0
        return t, np.where(t >= lead, 1 + movie._integral(r, t, lead), 0), r, t3

    rmax = float(b["max_rate"])
    n_end = run(rmax, 0.0)[1][-1]
    if n_end <= n_total:
        t, n, r, t3 = run(rmax, (n_total - n_end) / rmax)
    else:
        if run(r0, 0.0)[1][-1] > n_total:
            raise ValueError(f"build: {n_total:.0f} slices are too few for these times (shorten slow_s, ramp_s, stop_s)")
        lo, hi = r0, rmax
        for _ in range(60):
            mid = math.sqrt(lo * hi)
            lo, hi = (mid, hi) if run(mid, 0.0)[1][-1] < n_total else (lo, mid)
        t, n, r, t3 = run(lo, 0.0)
    scale = (n_total - 1) / (n[-1] - 1)                           # exactly n_total at the end
    return t, np.where(n > 0, 1 + (n - 1) * scale, 0), r * scale, t3


def timeline(m, b, fly, stack, end):
    """One row per frame: the build-up of ``stack`` from the fly-through's first slice to the plane
    ``end`` (part "build": slices imaged so far, the stack's planes lo and hi, the camera's look-at
    point lx, ly, lz, yaw and tilt (degrees) and distance, and how far the slice's rim has faded in),
    then the fly-through (part "fly": the rows of ``fly``, pipeline.movie.timeline's columns)."""
    w, h = m["size"]
    fps = m["fps"]
    up = -1 if m["view_rotation_deg"] == 0 else 1
    zoom0, cx0, cy0, z0 = (float(fly[k].iloc[0]) for k in ("zoom", "cx", "cy", "z"))
    n_total = round(end - z0)
    if n_total < 2:
        raise ValueError("build: the stack needs at least 2 slices")
    t, n, rate, t_stop = imaging(b, n_total)
    t_orbit = t_stop + b["orbit_delay_s"]
    n_frames = int(round((t_orbit + b["orbit_s"]) * fps))
    t_end = n_frames / fps                                      # the orbit ends on the fly-through's first frame
    if t_end < t[-1] - 1e-9:
        raise ValueError("build: the orbit must end after the imaging (orbit_delay_s + orbit_s >= stop_s)")
    if not 0 <= b["fade_s"] <= t_end - t_orbit:
        raise ValueError("build: fade_s must be between 0 and orbit_s")
    # Camera poses (look-at point, yaw, tilt, log distance): close to the first slice's corner nearest
    # the camera, the whole block, and face-on to the first slice at the fly-through's zoom.
    focal = w / 2 / math.tan(math.radians(b["fov_deg"]) / 2)
    yaw, tilt = (np.radians(b[k]) for k in ("yaw_deg", "tilt_deg"))
    first = stack.boxes[0]
    corner = np.array([first.x0 if math.sin(yaw[0]) < 0 else first.x1, first.y0 if up * tilt[0] < 0 else first.y1,
                       z0 + 1])
    target = np.array([w / 2 + up * w / 4, h / 2 + up * h / 4])  # a quarter in from the top left, on screen
    look = corner.copy()
    for _ in range(30):
        cam = movie.camera(look, yaw[0], tilt[0], focal * b["close_up"], up)
        miss = movie.project(corner[None], cam, (w, h), focal)[0] - target
        look += (miss[0] * cam[2] + miss[1] * cam[3]) * ((corner - cam[0]) @ cam[1]) / focal
    corners = stack.corners(z0, end)
    whole, dist = fit_view(corners, yaw[1], tilt[1], up, (w, h), focal, b["fill"])
    turn = 2 * math.pi * round(yaw[1] / (2 * math.pi))         # where the orbit ends: 0 or 360 degrees
    near = focal * b["close_up"]
    # Halfway round, the camera stands back as far as it must to keep the whole block in frame.
    far = fit_view(corners, (yaw[1] + turn) / 2, tilt[1] / 2, up, (w, h), focal, b["fill"])[1]
    bulge = max(0.0, math.log(far) - (math.log(dist) + math.log(focal * zoom0)) / 2)
    tf = np.arange(n_frames) / fps
    imaged = np.interp(tf, t, n)
    # The pull-back follows the slices imaged in log scale (10: slices it takes to get going), so the
    # stack grows steadily on screen and the camera comes to rest on the whole block with the last slice.
    # The look-at point's offset from the corner scales with the distance, so the corner glides on screen
    # from its place in the close-up to its place in the whole view.
    pull = (np.log1p(imaged / 10) / math.log1p(n_total / 10))[:, None]
    gap = near * (dist / near) ** pull
    drift = (corner + gap * ((1 - pull) * (look - corner) / near + pull * (whole - corner) / dist),
             yaw[0] + (yaw[1] - yaw[0]) * pull, tilt[0] + (tilt[1] - tilt[0]) * pull, np.log(gap))
    # The orbit (starting while the last slices come in) hands over to the face-on pose: at rest at the end.
    orbit = movie._ease(tf, t_orbit, t_end)[:, None]
    look, yaw, tilt, gap = (a * (1 - orbit) + f * orbit for a, f in
                            zip(drift, (np.array([cx0, cy0, z0]), turn, 0.0, math.log(focal * zoom0))))
    build = pd.DataFrame({"frame": np.arange(n_frames), "t": tf, "part": "build", "slices": np.floor(imaged + 1e-9),
                          "lo": z0, "hi": z0 + np.floor(imaged + 1e-9), "lx": look[:, 0], "ly": look[:, 1],
                          "lz": look[:, 2], "yaw": np.degrees(yaw[:, 0]), "tilt": np.degrees(tilt[:, 0]),
                          "dist": np.exp(gap[:, 0] + bulge * np.sin(np.pi * orbit[:, 0])),
                          "rim": movie._ease(tf, t_end - b["fade_s"], t_end)})
    fly = fly.assign(frame=fly["frame"] + n_frames, t=fly["t"] + t_end, part="fly")
    return pd.concat([build, fly], ignore_index=True)


def fit_view(corners, yaw, tilt, up, size, focal, fill):
    """(look-at point, distance) of a camera turned by ``yaw`` and raised by ``tilt`` (radians; see
    pipeline.movie.camera) that frames the points ``corners``: their picture centred, ``fill`` of the
    frame's height or width, whichever is tighter."""
    w, h = size
    centre, span = corners.mean(0), np.ptp(corners, 0).max()

    def picture(d):
        cam = movie.camera(centre, yaw, tilt, d, up)
        if ((corners - cam[0]) @ cam[1]).min() <= 0:                # behind the camera
            return None, np.inf
        p = movie.project(corners, cam, size, focal)
        return p, max(np.ptp(p[:, 0]) / (fill * w), np.ptp(p[:, 1]) / (fill * h))

    for _ in range(5):
        lo, hi = span * 1e-3, span * 1e3
        for _ in range(80):
            mid = math.sqrt(lo * hi)
            lo, hi = (mid, hi) if picture(mid)[1] > 1 else (lo, mid)
        p = picture(hi)[0]
        _, _, right, down = movie.camera(centre, yaw, tilt, hi, up)
        off = (p.min(0) + p.max(0)) / 2 - np.array([w / 2, h / 2])
        centre = centre + (off[0] * right + off[1] * down) * hi / focal
    return centre, hi


def rim_face(levels, size, zoom, cx, cy, z):
    """The whole slice ``z`` as pipeline.movie.render_frame frames it, as a face on the slice's -z side."""
    w, h = size
    img = movie.render_frame(levels, size, zoom, cx, cy, z)[0].astype(np.float32)
    return img, (cx - w / 2 * zoom, cy - h / 2 * zoom, z), (w * zoom, 0, 0), (0, h * zoom, 0), (0, 0, -1)


def render_build(stack, row, size, focal, up, rim=None):
    """Frame (uint8, unrotated) of a build-up row; ``rim`` (``rim_face``): the whole first slice, faded
    in by ``row.rim`` where the stack does not cover it."""
    cam = movie.camera((row.lx, row.ly, row.lz), math.radians(row.yaw), math.radians(row.tilt), row.dist, up)
    faces = stack.faces(row.lo, row.hi, cam, size, focal) if row.hi > row.lo else []
    out = movie.render_faces(faces, size, cam, focal, shade=0.0)
    if rim is not None and row.rim > 0:
        ones = [(np.ones(f[0].shape, np.float32), *f[1:]) for f in faces]
        cover = movie.render_faces(ones, size, cam, focal, shade=0.0)
        out += row.rim * movie.render_faces([rim], size, cam, focal, shade=0.0) * (1 - cover)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def load(cfg, levels):
    """(fly-through plan, Stack, end): the stack from the fly-through's first slice to the plane ``end``
    (after build.z_far), on the pixel grid of the fly-through's first frame."""
    m, b = cfg["movie"], cfg["build"]
    shape0 = levels[0].shape
    fly = fly_plan(m, b, shape0)
    zoom0, cx0, cy0, z0 = (float(fly[k].iloc[0]) for k in ("zoom", "cx", "cy", "z"))
    z_far = shape0[0] - 1 if b["z_far"] is None else int(b["z_far"])
    if not z0 < z_far < shape0[0]:
        raise ValueError(f"build.z_far must be between the fly-through's first slice {z0:.0f} and {shape0[0] - 1}")
    w, h = m["size"]
    stack = Stack(levels, footprints(levels, math.floor(z0), z_far + 1, b["tissue_fill"]), zoom0,
                  (cx0 - w / 2 * zoom0, cy0 - h / 2 * zoom0))
    if b["z_far"] is None and len(stack.boxes) > 1:             # up to the first change of footprint
        return fly, stack, z0 + math.floor(stack.boxes[1].za - stack.pad - z0)
    return fly, stack, z0 + math.floor(z_far + 1 - z0)


def make_frames(cfg, only=None, overwrite=False):
    """Write the PNG frames (all, or the frame numbers in ``only``); frames from an earlier run with
    the same timeline are kept, a changed timeline starts afresh."""
    m, b = cfg["movie"], cfg["build"]
    root = movie.volume_path(cfg)
    levels = movie.open_levels(root, m["cache_gb"])
    fly, stack, end = load(cfg, levels)
    plan = timeline(m, b, fly, stack, end)
    out = movie.frames_dir(cfg)
    plan_csv = out / "timeline.csv"
    settings = {k: m[k] for k in ("size", "scale_bar", "view_rotation_deg")} | {"volume": str(root)} | b \
        | {"boxes": [tuple(round(v, 3) for v in box) for box in stack.boxes]}
    plan_text = f"# {settings}\n" + plan.to_csv(index=False, float_format="%.6f")
    if overwrite or not plan_csv.exists() or plan_csv.read_text() != plan_text:
        for old in out.glob("f*.png"):
            old.unlink()
        cli.atomic_write(plan_csv, lambda p: Path(p).write_text(plan_text))
    build = plan[plan["part"] == "build"]
    nm = (serve.voxel_nm(root) or {"z": 8.0})["z"]
    z0 = build["lo"].iloc[0]
    log.info("%s: stack of slices %.0f-%.0f (%.1f um), footprint %s; build-up %.1f s (%d frames, up to %.0f "
             "slices/s), then the fly-through %.1f s", root.name, z0, end - 1, (end - z0) * nm / 1000,
             ", ".join(f"x {bx.x0:.0f}-{bx.x1:.0f} y {bx.y0:.0f}-{bx.y1:.0f} from z {bx.za:.0f}"
                       for bx in stack.boxes if bx.za < end),
             len(build) / m["fps"], len(build), np.diff(build["slices"]).max() * m["fps"], len(fly) / m["fps"])
    w, h = m["size"]
    focal = w / 2 / math.tan(math.radians(b["fov_deg"]) / 2)
    up = -1 if m["view_rotation_deg"] == 0 else 1             # the y direction that is up on screen
    voxel = serve.voxel_nm(root) or {"x": 8.0}
    f0 = fly.iloc[0]
    rim = None
    rows = plan if only is None else plan[plan["frame"].isin(only)]
    done = 0
    for row in rows.itertuples():
        path = out / f"f{row.frame:05d}.png"
        if path.exists():
            continue
        if row.part == "build":
            if rim is None and row.rim > 0:
                rim = rim_face(levels, m["size"], f0.zoom, f0.cx, f0.cy, f0.z)
            frame = render_build(stack, row, m["size"], focal, up, rim)
        else:
            frame = movie.render_frame(levels, m["size"], row.zoom, row.cx, row.cy, row.z, row.rotation)[0]
        if m["view_rotation_deg"] == 180:
            frame = np.ascontiguousarray(np.rot90(frame, 2))
        if m["scale_bar"] and row.part == "fly":
            frame = movie.draw_scale_bar(frame, row.zoom * voxel["x"])
        cli.atomic_write(path, lambda p: cv2.imwrite(p, frame, [cv2.IMWRITE_PNG_COMPRESSION, 1]))
        done += 1
        if done % 50 == 0:
            log.info("frame %d/%d (%s)", row.frame + 1, len(plan),
                     f"{row.slices:.0f} slices" if row.part == "build" else f"zoom {row.zoom:.2f}, z {row.z:.1f}")
    log.info("%d frames written to %s", done, out)
    return out


def encode(cfg):
    plan_csv = movie.frames_dir(cfg) / "timeline.csv"
    if not plan_csv.exists():
        raise RuntimeError("no timeline: run 'frames' first")
    n = len(pd.read_csv(plan_csv, comment="#"))
    missing = [k for k in range(n) if not (movie.frames_dir(cfg) / f"f{k:05d}.png").exists()]
    if missing:
        raise RuntimeError(f"{len(missing)} of {n} frames missing (first: {missing[0]}): run 'frames' first")
    cmd = movie.encode_command(cfg)
    subprocess.run(cmd, check=True)
    log.info("wrote %s", cmd[-1])


def main(argv=None):
    p = cli.base_parser(" ".join(__doc__.split("\n\n")[0].split()))
    p.add_argument("action", choices=("frames", "encode"))
    p.add_argument("--frames", help="frames: only these frame numbers, e.g. 0,240,480 (a quick look)")
    p.add_argument("--print", action="store_true", help="encode: print the ffmpeg command instead of running it")
    args = p.parse_args(argv)
    cfg = cli.setup(args, deep_merge(movie.DEFAULTS, DEFAULTS))
    if args.action == "frames":
        only = [int(v) for v in args.frames.split(",")] if args.frames else None
        make_frames(cfg, only, args.overwrite)
    elif args.print:
        print(shlex.join(movie.encode_command(cfg)))
    else:
        if not shutil.which(cfg["movie"]["ffmpeg"]):
            raise FileNotFoundError(f"{cfg['movie']['ffmpeg']} not found (an FFmpeg module? EM_PIPE_FFMPEG_MODULE in slurm/site.sh)")
        encode(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
