"""Segment meshes revealed by the scroll in the tissue movie's fly-through (``tissue_movie.segments``).

At full zoom (in the fly-through's hold, ``movie.hold_end_s``) the scroll speeds up (``slices_per_s``) and
every slice it passes leaves the chosen segments' part of it behind as their meshes: from the plane where
the reveal started to the cut, the tissue the scroll has milled away. Meanwhile the camera turns
(``yaw_deg``, ``tilt_deg``) and pulls back (``pull``) and the point it looks at falls back to halfway
between that plane and the cut (it moves at half the scroll's speed): the section recedes and the
segments stand in front of it. The scroll slows back down and the segments hold; then (``hide``) the
scroll reverses (``back_slices_per_s``) and the returning section swallows them slice by slice while the
camera comes face-on again (``fall_s``), until the cut is back at the plane where the reveal started as
the zoom-out begins, its own scroll-back carrying on from there; or they fade out (``fade_s``) before the
zoom-out. Either way the edges of a segmentation cutout never show as the view widens.

The meshes come from a Neuroglancer precomputed segmentation (``source``, e.g. precomputed://https://...;
read with CloudVolume, no credentials) in the volume's own frame (nm, as the volume's voxel size makes s0
voxels). Blender renders them, lit, on a transparent background, with the same camera as each frame
(``blender_camera``), and pipeline.movie_tissue lays that layer over the frame. Nothing is in front of the
meshes (the volume in front of the cut is cut away), so no depth compositing is needed.

    python -m pipeline.movie_tissue --config C meshes    # fetch the meshes, render their layers (Blender)
    python -m pipeline.movie_tissue --config C frames    # the frames, with the layers on them
"""

import hashlib
import json
import logging
import math
import os
import re
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import cli, serve
from .movie import camera, ease

log = logging.getLogger(__name__)

DEFAULTS = {
    "source": None,       # precomputed segmentation with meshes, in the volume's frame: precomputed://https://...
    "ids": [],            # segment IDs shown (none: no reveal)
    "colors": None,       # [r, g, b] (0-1) per segment; null: PALETTE
    "start_s": 0.5,       # the reveal starts this long after the fly-through's zoom ends (in movie.hold_end_s)
    "slices_per_s": 500,  # the scroll's speed while it reveals ...
    "speed_s": 1.0,       # ... reached over this long (and back to the fly-through's over this long) ...
    "reveal_s": 5.0,      # ... and kept this long; the rest of movie.hold_end_s at the fly-through's speed
    "rise_s": 2.0,        # the camera turns and pulls back over this long from the reveal's start ...
    "fall_s": 2.5,        # ... and comes face-on again over this long: ending as the zoom-out starts (hide:
                          # scroll), or from its start (fade)
    "hide": "scroll",     # how the meshes go before the zoom-out: scroll (the scroll reverses and the section
                          # swallows them: the hold ends as late as that allows) or fade (fade_s)
    "back_slices_per_s": None,   # scroll: the scroll's speed backwards (reached over speed_s); null: slices_per_s
    "fade_s": 1.0,        # fade: the meshes fade out over this long, ending as the zoom-out starts
    "yaw_deg": 25,        # the camera's turn about the y axis ...
    "tilt_deg": 12,       # ... its rise towards up on screen ...
    "pull": 1.8,          # ... and its distance (x the face-on distance), turned
    # The look (Blender, Cycles)
    "roughness": 0.45,    # the meshes' surface (Principled BSDF)
    "subsurface": 0.0,    # subsurface scattering weight (1: waxy, light passes into the mesh) ...
    "subsurface_scale": 30,   # ... and how far (s0 voxels)
    "hdri": None,         # an equirectangular HDR image lighting the meshes (no lamps); null: a grey sky, a key
                          # and a fill light moving with the camera
    "hdri_rotation_deg": [0, 0, 0],   # its rotation (Blender's mapping, XYZ Euler) with its up on screen up
    "hdri_strength": 1.0,
    "exposure": 1.0,      # film exposure (a multiplier)
    "samples": 32,        # Blender (Cycles) samples per px, denoised
    "blender": None,      # the Blender executable; null: $EM_PIPE_BLENDER, else blender
}

# Distinct, saturated colours (sRGB, 0-1), in order.
PALETTE = [(0.90, 0.25, 0.30), (0.20, 0.60, 0.95), (0.98, 0.75, 0.15), (0.35, 0.80, 0.35), (0.75, 0.40, 0.90),
           (0.10, 0.80, 0.80), (0.98, 0.50, 0.20), (0.95, 0.45, 0.75), (0.55, 0.85, 0.15), (0.30, 0.35, 0.95),
           (0.85, 0.85, 0.85), (0.60, 0.35, 0.20)]


# The settings that only time the reveal: they change the timeline's rows, not how a row looks.
TIMING = ("start_s", "slices_per_s", "speed_s", "reveal_s", "rise_s", "fall_s", "hide", "back_slices_per_s",
          "fade_s", "blender")


def enabled(m):
    return bool(m["segments"]["ids"])


def look(s):
    """The segments' settings that decide how a frame with meshes looks, given its timeline row."""
    return {k: v for k, v in s.items() if k not in TIMING}


def add_reveal(fly, mv, s):
    """The fly-through's timeline (pipeline.movie.timeline, no hold at the start; zoom-out at its end) with
    the reveal: the scroll sped up to slices_per_s (z) and, hide scroll, reversed to swallow the meshes
    before the zoom-out; the camera's turn ``reveal`` (0-1: up over rise_s from the reveal's start, down over
    fall_s: ending as the zoom-out starts, or for hide fade from then); ``mesh_from``, the plane where the
    reveal started (the cut then; NaN before it: the meshes show between it and the cut); ``mesh_alpha``,
    their opacity (hide fade: fading out over fade_s before the zoom-out)."""
    t = fly["t"].to_numpy()
    t1, t4 = mv["zoom_s"], mv["zoom_s"] + mv["hold_end_s"]
    r0 = t1 + s["start_s"]
    r1 = r0 + s["speed_s"] + s["reveal_s"]
    if min(s["start_s"], s["speed_s"], s["reveal_s"], s["fade_s"]) < 0 or min(s["rise_s"], s["fall_s"],
                                                                         s["slices_per_s"]) <= 0:
        raise ValueError("tissue_movie.segments: the times must be >= 0, rise_s, fall_s and slices_per_s > 0")
    if s["hide"] not in ("scroll", "fade"):
        raise ValueError(f"tissue_movie.segments.hide must be scroll or fade, not {s['hide']!r}")
    if mv["direction"] != 1:
        raise ValueError("tissue_movie.segments: needs movie.direction 1 (the scroll towards higher z)")
    if s["hide"] == "fade" and s["fade_s"] and t4 - s["fade_s"] <= r0:
        raise ValueError("tissue_movie.segments: fade_s must start after the reveal does (a longer movie.hold_end_s)")
    if max(r1 + s["speed_s"], r0 + s["rise_s"]) > t4 + 1e-9:
        raise ValueError(f"tissue_movie.segments: the reveal needs movie.hold_end_s >= "
                         f"{s['start_s'] + max(2 * s['speed_s'] + s['reveal_s'], s['rise_s']):g}")
    z = fly["z"].to_numpy()
    v = np.gradient(z, t) if len(t) > 1 else np.zeros_like(z)
    fast = ease(t, r0, r0 + s["speed_s"]) * (1 - ease(t, r1, r1 + s["speed_s"]))
    extra = (np.sign(mv["direction"]) * s["slices_per_s"] - v) * fast
    out = fly.copy()
    out["z"] = z + _integral(extra, t)
    start = float(np.interp(r0, t, out["z"])) + 0.5
    if s["hide"] == "fade":
        out["reveal"] = ease(t, r0, r0 + s["rise_s"]) * (1 - ease(t, t4, t4 + s["fall_s"]))
        out["mesh_alpha"] = 1 - ease(t, t4 - s["fade_s"], t4) if s["fade_s"] else np.ones(len(t))
    else:
        out["z"] = out["z"] + _back(t, v, s, r1 + s["speed_s"], t4, out["z"].to_numpy(), start - 1)
        out["reveal"] = ease(t, r0, r0 + s["rise_s"]) * (1 - ease(t, t4 - s["fall_s"], t4))
        out["mesh_alpha"] = 1.0
    out["mesh_from"] = np.where(t >= r0, start, np.nan)
    return out


def _integral(v, t):
    """The integral of ``v`` (sampled at times ``t``) from t[0] on (trapezoids)."""
    return np.concatenate([[0], np.cumsum((v[1:] + v[:-1]) / 2 * np.diff(t))])


def _back(t, v, s, earliest, t4, z, z_end):
    """The change to ``z`` (the forward scroll, speed ``v``) by the reverse scroll that ends at ``t4`` (the
    zoom-out) with the cut's slice at ``z_end``: from ``tb`` the speed eases from v to -back_slices_per_s
    over speed_s, holds, and eases to -v over the last speed_s (the zoom-out's scroll-back takes over from
    there); tb as late as that allows (the longest hold), and not before ``earliest`` (the reveal's scroll)."""
    back = float(s["back_slices_per_s"] or s["slices_per_s"])
    sp = s["speed_s"]

    def change(tb):
        w = (-back - v) * ease(t, tb, tb + sp) * (1 - ease(t, t4 - sp, t4)) - 2 * v * ease(t, t4 - sp, t4)
        return _integral(w, t)

    def end(tb):                                      # the cut's slice at t4
        return float(np.interp(t4, t, z + change(tb)))

    lo, hi = earliest, t4 - 2 * sp
    if hi < lo or end(lo) > z_end:
        raise ValueError("tissue_movie.segments: the reverse scroll (hide: scroll) cannot swallow the meshes before "
                         "the zoom-out: a longer movie.hold_end_s or a faster back_slices_per_s")
    for _ in range(60):                               # end(tb) grows with tb: the latest tb that reaches z_end
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if end(mid) <= z_end else (lo, mid)
    return change(lo)


def shown(z, mesh_from):
    """The meshes' z range (lo, hi: the reveal's start plane, the cut z + 0.5) at a frame, or None."""
    if mesh_from is None or not np.isfinite(mesh_from) or z + 0.5 <= mesh_from:
        return None
    return float(mesh_from), float(z + 0.5)


def view(s, reveal):
    """(yaw, tilt in radians, distance factor) at ``reveal`` (0-1)."""
    r = float(reveal) if reveal is not None and np.isfinite(reveal) else 0.0
    return math.radians(s["yaw_deg"]) * r, math.radians(s["tilt_deg"]) * r, 1 + (s["pull"] - 1) * r


def fly_camera(s, zoom, cx, cy, z, reveal, mesh_from, focal, up):
    """The fly-through's camera at a frame: face-on to the cut z + 0.5 at the distance that shows it at
    ``zoom``; by the ``reveal``, turned, raised and pulled back about the point it looks at, which falls back
    from the cut to halfway between it and the plane ``mesh_from`` (the middle of the meshes shown)."""
    yaw, tilt, pull = view(s, reveal)
    span = shown(z, mesh_from)
    r = float(reveal) if reveal is not None and np.isfinite(reveal) else 0.0
    look = z + 0.5 - (r * (span[1] - span[0]) / 2 if span else 0.0)
    return camera((cx, cy, look), yaw, tilt, focal * zoom * pull, up)


def blender_camera(cam, size, focal, origin):
    """The Blender camera for ``cam`` (pipeline.movie.camera): its world matrix (4 x 4, rows; the world
    shifted by -``origin``) and lens (mm) on a 36 mm sensor fitted to the frame's long side. Blender's
    camera looks along its -Z with +Y up: columns right, -down, -forward (a rotation: right x down =
    forward)."""
    pos, forward, right, down = (np.asarray(v, float) for v in cam)
    mat = np.eye(4)
    mat[:3, 0], mat[:3, 1], mat[:3, 2] = right, -down, -forward
    mat[:3, 3] = pos - np.asarray(origin, float)
    return {"matrix": mat.tolist(), "lens": focal * 36.0 / max(size), "sensor": 36.0}


def blender_project(bcam, size, point):
    """Frame px of a (shifted) world point through a ``blender_camera`` (a pinhole, as Blender's)."""
    mat = np.asarray(bcam["matrix"])
    q = mat[:3, :3].T @ (np.asarray(point, float) - mat[:3, 3])        # camera space: x right, y up, -z ahead
    f = bcam["lens"] / bcam["sensor"] * max(size)
    return np.array([size[0] / 2 + f * q[0] / -q[2], size[1] / 2 - f * q[1] / -q[2]])


# ---------------------------------------------------------------------------------------------------
# Meshes


def mesh_dir(work):
    return Path(work) / "segments"


def write_ply(path, vertices, faces):
    """Binary little-endian PLY of float32 vertices and triangle faces."""
    v = np.ascontiguousarray(vertices, "<f4")
    f = np.empty(len(faces), [("n", "u1"), ("i", "<i4", (3,))])
    f["n"], f["i"] = 3, faces
    head = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(v)}\nproperty float x\nproperty float y\n"
            f"property float z\nelement face {len(f)}\nproperty list uchar int vertex_indices\nend_header\n")

    def write(p):
        with open(p, "wb") as fh:
            fh.write(head.encode())
            fh.write(v.tobytes())
            fh.write(f.tobytes())
    cli.atomic_write(path, write)


FETCH_THREADS = 8   # meshes fetched at once (each a few requests, ~3 s)


def fetch(s, volume, work):
    """The segments' meshes as PLY files in s0 voxels (x, y, z) of ``volume`` in ``work``/segments/; those
    already there are kept. FETCH_THREADS at a time, one CloudVolume per thread."""
    from cloudvolume import CloudVolume

    out = mesh_dir(work)
    out.mkdir(parents=True, exist_ok=True)
    voxel = serve.voxel_nm(volume)
    if not voxel:
        raise ValueError(f"{volume}: no voxel size in its metadata")
    scale = np.array([voxel["x"], voxel["y"], voxel["z"]], float)
    todo = [int(sid) for sid in s["ids"] if not (out / f"{int(sid)}.ply").exists()]
    local = threading.local()

    def one(sid):
        if not hasattr(local, "cv"):
            local.cv = CloudVolume(s["source"], use_https=True, progress=False)
        mesh = local.cv.mesh.get(sid)
        mesh = mesh[sid] if isinstance(mesh, dict) else mesh
        write_ply(out / f"{sid}.ply", mesh.vertices / scale, mesh.faces)
        return sid, len(mesh.vertices), len(mesh.faces)

    if todo:
        with ThreadPoolExecutor(min(FETCH_THREADS, len(todo))) as pool:
            for sid, nv, nf in pool.map(one, todo):
                log.info("segment %d: %d vertices, %d faces", sid, nv, nf)
    return [out / f"{int(sid)}.ply" for sid in s["ids"]]


def colors(s):
    c = s["colors"] or [PALETTE[i % len(PALETTE)] for i in range(len(s["ids"]))]
    if len(c) < len(s["ids"]):
        raise ValueError("tissue_movie.segments.colors: one per segment")
    return [list(map(float, v)) for v in c]


def layer_rows(plan):
    """The timeline rows (a boolean mask) whose frames show meshes: fly-through rows whose cut is past the
    reveal's start plane, the meshes not faded out."""
    if "mesh_from" not in plan.columns:
        return np.zeros(len(plan), bool)
    return ((plan["part"] == "fly") & plan["mesh_from"].notna() & (plan["z"] + 0.5 > plan["mesh_from"])
            & (plan["mesh_alpha"] > 0)).to_numpy()


def plan_layers(s, rows, out, size, focal, up, origin, overwrite=False):
    """(the Blender frames still to render, {frame: key} of those) for the mesh layers of timeline ``rows``
    in ``out``. A layer is kept if the look (``look``, size, origin) is unchanged and its frame's camera and
    clip are the same: its key, in m<frame>.key next to it (one file per layer, so jobs rendering different
    frames side by side don't overwrite each other's), or for older layers in out/frames.json or the last
    job.json. A layer a job rendered but did not record (cancelled before the end) counts if its PNG is
    complete and the newest job file listing it has its key. A frame whose view a layer of another frame
    already shows (the same key: e.g. the timeline moved) gets a copy of it. A changed look removes every
    layer."""
    out = Path(out)
    stamp, record = out / "settings.json", out / "frames.json"
    text = json.dumps(look(s) | {"size": list(size), "origin": list(origin)}, sort_keys=True)
    keys = json.loads(record.read_text()) if record.exists() else {}
    if not record.exists() and (out / "job.json").exists() and stamp.exists():
        old = json.loads(stamp.read_text())               # (the settings stamp before only the look was in it)
        if json.dumps({k: old.get(k) for k in json.loads(text)}, sort_keys=True) == text:
            keys = {re.search(r"m(\d+)\.png$", f["path"]).group(1).lstrip("0") or "0": _key(f)
                    for f in json.loads((out / "job.json").read_text())["frames"]}
            stamp.write_text(text)
    if overwrite or not stamp.exists() or stamp.read_text() != text:
        for old in [*out.glob("m*.png"), *out.glob("m*.key"), *out.glob("job*.json"), record]:
            old.unlink(missing_ok=True)
        keys = {}
        cli.atomic_write(stamp, lambda p: Path(p).write_text(text))
    known = _known_layers(out, keys)                  # {layer PNG: key}
    have = {}                                         # {key: a layer PNG with it}
    for png, key in known.items():
        have.setdefault(key, png)
    frames, new = [], {}
    for row in rows.itertuples():
        path = out / f"m{row.frame:05d}.png"
        cam = fly_camera(s, row.zoom, row.cx, row.cy, row.z, row.reveal, row.mesh_from, focal, up)
        lo, hi = shown(row.z, row.mesh_from)
        frame = {"path": str(path), "camera": blender_camera(cam, size, focal, origin),
                 "clip": [lo - origin[2], hi - origin[2]]}
        key = _key(frame)
        if known.get(path) == key:
            if not path.with_suffix(".key").exists():
                record_keys(out, {row.frame: key})
            continue
        if key in have:                               # the same view as another frame's layer: a copy (not a link:
            cli.atomic_write(path, lambda p, src=have[key]: shutil.copyfile(src, p))   # Blender may rewrite either)
            record_keys(out, {row.frame: key})
            continue
        frames.append(frame)
        new[row.frame] = key
    return frames, new


def _known_layers(out, legacy):
    """{layer PNG: key} of the layers in ``out``: from their m<frame>.key; for PNGs without one, from
    ``legacy`` ({frame: key}: frames.json, the first job.json) or else the newest job file listing them
    that is not newer than the PNG, if the PNG is complete (a job cancelled before recording its layers)."""
    known = {}
    for side in out.glob("m*.key"):
        if side.with_suffix(".png").exists():
            known[side.with_suffix(".png")] = side.read_text()
    listed = {}
    for job in sorted(out.glob("job*.json"), key=lambda p: p.stat().st_mtime):
        try:
            entries = json.loads(job.read_text())["frames"]
        except (OSError, ValueError, KeyError):
            continue
        for f in entries:
            listed[Path(f["path"]).name] = (job.stat().st_mtime, _key(f))
    for png in out.glob("m*.png"):
        if png in known:
            continue
        frame = png.stem[1:].lstrip("0") or "0"
        if frame in legacy:
            known[png] = legacy[frame]
        elif png.name in listed and listed[png.name][0] <= png.stat().st_mtime + 1 and _complete(png):
            known[png] = listed[png.name][1]
    return known


def _complete(png):
    """Whether a PNG decodes (one cut short by a cancelled job does not)."""
    import cv2

    return cv2.imread(str(png), cv2.IMREAD_UNCHANGED) is not None


def record_keys(out, keys):
    """Record each layer's key ({frame: key}) in m<frame>.key next to it."""
    for frame, key in keys.items():
        cli.atomic_write(Path(out) / f"m{int(frame):05d}.key", lambda p, key=key: Path(p).write_text(key))


def _key(frame):
    """A layer's camera and clip, as a short digest."""
    view = json.dumps({"camera": frame["camera"], "clip": frame["clip"]}, sort_keys=True)
    return hashlib.sha1(view.encode()).hexdigest()[:16]


def render_layers(cfg, plan, frames_dir, volume, work, size, focal, up, overwrite=False, only=None):
    """Render the mesh layers (RGBA PNG, ``frames_dir``/meshes/m<frame>.png) of the frames that show meshes
    (``layer_rows``; of those, the frame numbers in ``only``), with Blender; layers from an earlier run
    with the same look, camera and clip are kept (``plan_layers``)."""
    s = cfg["tissue_movie"]["segments"]
    paths = fetch(s, volume, work)
    rows = plan[layer_rows(plan)]
    if only is not None:
        rows = rows[rows["frame"].isin(only)]
    out = Path(frames_dir) / "meshes"
    out.mkdir(parents=True, exist_ok=True)
    origin = [float(v) for v in cfg["movie"]["target"]]
    frames, keys = plan_layers(s, rows, out, size, focal, up, origin, overwrite)
    if not frames:
        log.info("mesh layers: all %d done", len(rows))
        return out
    look = ("samples", "roughness", "subsurface", "subsurface_scale", "hdri", "hdri_rotation_deg", "hdri_strength",
            "exposure")
    job = {"size": list(size), "origin": origin, "frames": frames, **{k: s[k] for k in look},
           "meshes": [{"path": str(p), "color": c} for p, c in zip(paths, colors(s))]}
    job_path = out / f"job-{min(keys):05d}-{max(keys):05d}.json"     # (one per job: jobs may run side by side)
    cli.atomic_write(job_path, lambda p: Path(p).write_text(json.dumps(job)))
    exe = s["blender"] or os.environ.get("EM_PIPE_BLENDER") or "blender"
    if not shutil.which(exe):
        raise FileNotFoundError(f"{exe} not found: set tissue_movie.segments.blender or EM_PIPE_BLENDER (slurm/site.sh)")
    script = Path(__file__).with_name("blender_segments.py")
    log.info("mesh layers: %d of %d frames to render with %s", len(frames), len(rows), exe)
    subprocess.run([exe, "-b", "--factory-startup", "--python", str(script), "--", str(job_path)], check=True)
    missing = [f["path"] for f in frames if not Path(f["path"]).exists()]
    if missing:
        raise RuntimeError(f"Blender left {len(missing)} mesh layers unwritten, e.g. {missing[0]}")
    record_keys(out, keys)
    return out
