"""Fly-through movie of a rendered volume: the whole slice first, then a smooth zoom into a target
point down to full resolution, scrolling through z the whole time (faster when zoomed out, if set);
optionally the cutting plane then turns about the frame's vertical axis (to the y-z plane at 90
degrees, showing the voxels are isotropic) while the scroll carries on along the plane's normal,
and finally zooms back out until the whole plane (e.g. the side view, y by z) fills the frame.

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
    z = an exact slice) and the cutting plane's rotation about the frame's vertical axis (degrees)."""
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
    t3 = t2 + (m["rotate_s"] if m["rotate_deg"] else 0)
    t4 = t3 + (m["rotated_s"] if m["rotate_deg"] else 0)
    t5 = t4 + m["zoom_out_s"]
    if min(t0, m["hold_end_s"], m["rotate_s"], m["rotated_s"], m["zoom_out_s"], m["hold_out_s"]) < 0 \
            or m["zoom_s"] <= 0:
        raise ValueError("movie: the hold, rotation and zoom-out times must be >= 0 and zoom_s > 0")
    side = abs(abs(m["rotate_deg"]) - 90) < 1e-9
    if m["zoom_out_s"] and not (side or m["rotate_deg"] == 0):
        raise ValueError("movie: zoom_out_s needs rotate_deg 0, 90 or -90")
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
    theta = np.radians(m["rotate_deg"]) * _ease(t, t2, t3) if m["rotate_deg"] else np.zeros(n)
    # The view moves along the cutting plane's normal, (-sin, 0, cos) in (x, y, z), and passes the
    # target slice when the zoom ends (t1).
    d = m["direction"]
    cx = cx + _integral(-d * speed * np.sin(theta), t, t1)
    z = tz + _integral(d * speed * np.cos(theta), t, t1)
    if m["zoom_out_s"]:
        # The zoom-out mirrors the zoom-in: from the view at t4 straight to the final plane's centre.
        ko = (zoom_out - ze) / (zo - ze) if zo != ze else np.zeros(n)
        cy = cy + (ny / 2 - np.interp(t4, t, cy)) * ko
        if side:
            z = z + ((nz - 1) / 2 - np.interp(t4, t, z)) * ko
        else:
            cx = cx + (nx / 2 - np.interp(t4, t, cx)) * ko
    # Up to the zoom-out every frame must lie in the volume (a turned frame spans w/2 * zoom in z).
    upto = t <= t4 + 1e-9
    span = np.where(upto, w / 2 * zoom * np.abs(np.sin(theta)), 0.0)
    if (z - span).min() < -1e-6 or (z + span).max() > nz - 1 + 1e-6:
        lo, hi = (z - span - tz).min(), (z + span - tz).max()
        raise ValueError(f"movie: the frames reach from {lo:+.0f} to {hi:+.0f} slices around the target z, "
                         f"outside the volume's 0-{nz - 1}: target z must be between "
                         f"{math.ceil(-lo - 1e-6)} and {math.floor(nz - 1 - hi + 1e-6)} "
                         f"(or shorten the movie, or scroll slower)")
    return pd.DataFrame({"frame": np.arange(n), "t": t, "zoom": zoom, "cx": cx, "cy": cy, "z": z,
                         "rotation": np.degrees(theta)})


def speeds(m):
    """(zoomed-out, zoomed-in) scroll speed in slices/s."""
    s = m["slices_per_s"]
    return (float(s[0]), float(s[1])) if isinstance(s, (list, tuple)) else (float(s), float(s))


def movie_seconds(m):
    rotation = m["rotate_s"] + m["rotated_s"] if m["rotate_deg"] else 0
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
    uc), rows along y (centre vc); axis "x": the plane x = cut, columns along z, rows along y.
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
    n_cut = arr.shape[0] if axis == "z" else arr.shape[2]
    cl = min(max(cut / f - 0.5, 0.0), n_cut - 1.0)    # the cut on this level (between two planes)
    ca = int(math.floor(cl))
    frac = cl - ca
    two = frac > 1e-3 and ca + 1 < n_cut
    if axis == "z":
        planes = _read(arr, ca, ca + 1 + two, v0, v1, u0, u1)
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


STRIP = 128   # frame columns per read in _turned


def _turned(levels, size, zoom, cx, cy, z, theta):
    """A frame on the plane through (cx, cy, z) that contains the y axis and is turned by ``theta``
    (radians) from the x-y plane: frame columns run along (cos, 0, sin) in (x, y, z). Trilinear
    sampling; no anti-aliasing, so meant for zooms up to ~1.5 voxels per frame px."""
    w, h = size
    level = _level(levels, zoom)
    arr, f = levels[level], 2 ** level
    s = (np.arange(w) + 0.5 - w / 2) * zoom                       # s0 voxels from the centre
    xs = (cx + s * math.cos(theta)) / f - 0.5                     # level pixel indices
    zs = (z + 0.5 + s * math.sin(theta)) / f - 0.5
    ys = (cy + (np.arange(h) + 0.5 - h / 2) * zoom) / f - 0.5
    ya, yb = math.floor(ys[0]), math.floor(ys[-1]) + 2
    iy = (np.floor(ys).astype(int) - ya)[None, :]
    wy = (ys - np.floor(ys)).astype(np.float32)[None, :]
    out = np.empty((h, w), np.float32)
    for c0 in range(0, w, STRIP):
        x, zz = xs[c0:c0 + STRIP], zs[c0:c0 + STRIP]
        xa, za = math.floor(x.min()), math.floor(zz.min())
        box = _read(arr, za, math.floor(zz.max()) + 2, ya, yb, xa, math.floor(x.max()) + 2)
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
    settings = {k: m[k] for k in ("size", "scale_bar", "view_rotation_deg")} | {"volume": str(root)}
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
    done = 0
    for row in rows.itertuples():
        path = out / f"f{row.frame:05d}.png"
        if path.exists():
            continue
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
