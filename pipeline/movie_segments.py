"""Segment meshes revealed by the scroll in the tissue movie's fly-through (``tissue_movie.segments``).

At full zoom (in the fly-through's hold, ``movie.hold_end_s``) the scroll speeds up (``slices_per_s``) and
every slice it passes leaves the chosen segments' part of it behind as their meshes: from the plane where
the reveal started to the cut, the tissue the scroll has milled away. Meanwhile the camera turns
(``yaw_deg``, ``tilt_deg``) and pulls back (``pull``) and the point it looks at falls back to halfway
between that plane and the cut (it moves at half the scroll's speed): the section recedes and the
segments stand in front of it. The scroll slows back down and the segments hold; with the zoom-out the
scroll reverses and the returning section swallows them, as the camera comes face-on again (``fall_s``).

The meshes come from a Neuroglancer precomputed segmentation (``source``, e.g. precomputed://https://...;
read with CloudVolume, no credentials) in the volume's own frame (nm, as the volume's voxel size makes s0
voxels). Blender renders them, lit, on a transparent background, with the same camera as each frame
(``blender_camera``), and pipeline.movie_tissue lays that layer over the frame. Nothing is in front of the
meshes (the volume in front of the cut is cut away), so no depth compositing is needed.

    python -m pipeline.movie_tissue --config C meshes    # fetch the meshes, render their layers (Blender)
    python -m pipeline.movie_tissue --config C frames    # the frames, with the layers on them
"""

import json
import logging
import math
import os
import shutil
import subprocess
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
    "fall_s": 2.5,        # ... and comes face-on again over this long from the zoom-out's start
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


def enabled(m):
    return bool(m["segments"]["ids"])


def add_reveal(fly, mv, s):
    """The fly-through's timeline (pipeline.movie.timeline, no hold at the start; zoom-out at its end) with
    the reveal: the scroll sped up to slices_per_s (z), the camera's turn ``reveal`` (0-1: up over rise_s
    from the reveal's start, down over fall_s from the zoom-out's start) and ``mesh_from``, the plane where
    the reveal started (the cut then; NaN before it): the meshes show between it and the cut."""
    t = fly["t"].to_numpy()
    t1, t4 = mv["zoom_s"], mv["zoom_s"] + mv["hold_end_s"]
    r0 = t1 + s["start_s"]
    r1 = r0 + s["speed_s"] + s["reveal_s"]
    if min(s["start_s"], s["speed_s"], s["reveal_s"]) < 0 or min(s["rise_s"], s["fall_s"], s["slices_per_s"]) <= 0:
        raise ValueError("tissue_movie.segments: the times must be >= 0, rise_s, fall_s and slices_per_s > 0")
    if max(r1 + s["speed_s"], r0 + s["rise_s"]) > t4 + 1e-9:
        raise ValueError(f"tissue_movie.segments: the reveal needs movie.hold_end_s >= "
                         f"{s['start_s'] + max(2 * s['speed_s'] + s['reveal_s'], s['rise_s']):g}")
    z = fly["z"].to_numpy()
    v = np.gradient(z, t) if len(t) > 1 else np.zeros_like(z)
    fast = ease(t, r0, r0 + s["speed_s"]) * (1 - ease(t, r1, r1 + s["speed_s"]))
    extra = (np.sign(mv["direction"]) * s["slices_per_s"] - v) * fast
    out = fly.copy()
    out["z"] = z + np.concatenate([[0], np.cumsum((extra[1:] + extra[:-1]) / 2 * np.diff(t))])
    out["reveal"] = ease(t, r0, r0 + s["rise_s"]) * (1 - ease(t, t4, t4 + s["fall_s"]))
    out["mesh_from"] = np.where(t >= r0, float(np.interp(r0, t, out["z"])) + 0.5, np.nan)
    return out


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


def fetch(s, volume, work):
    """The segments' meshes as PLY files in s0 voxels (x, y, z) of ``volume`` in ``work``/segments/; those
    already there are kept."""
    from cloudvolume import CloudVolume

    out = mesh_dir(work)
    out.mkdir(parents=True, exist_ok=True)
    voxel = serve.voxel_nm(volume)
    if not voxel:
        raise ValueError(f"{volume}: no voxel size in its metadata")
    scale = np.array([voxel["x"], voxel["y"], voxel["z"]], float)
    cv = None
    for sid in s["ids"]:
        path = out / f"{int(sid)}.ply"
        if path.exists():
            continue
        if cv is None:
            cv = CloudVolume(s["source"], use_https=True, progress=False)
        mesh = cv.mesh.get(int(sid))
        mesh = mesh[int(sid)] if isinstance(mesh, dict) else mesh
        write_ply(path, mesh.vertices / scale, mesh.faces)
        log.info("segment %d: %d vertices, %d faces", sid, len(mesh.vertices), len(mesh.faces))
    return [out / f"{int(sid)}.ply" for sid in s["ids"]]


def colors(s):
    c = s["colors"] or [PALETTE[i % len(PALETTE)] for i in range(len(s["ids"]))]
    if len(c) < len(s["ids"]):
        raise ValueError("tissue_movie.segments.colors: one per segment")
    return [list(map(float, v)) for v in c]


def layer_rows(plan):
    """The timeline rows (a boolean mask) whose frames show meshes: fly-through rows whose cut is past the
    reveal's start plane."""
    if "mesh_from" not in plan.columns:
        return np.zeros(len(plan), bool)
    return ((plan["part"] == "fly") & plan["mesh_from"].notna() & (plan["z"] + 0.5 > plan["mesh_from"])).to_numpy()


def render_layers(cfg, plan, frames_dir, volume, work, size, focal, up, overwrite=False):
    """Render the mesh layers (RGBA PNG, ``frames_dir``/meshes/m<frame>.png) of the frames that show meshes
    (``layer_rows``), with Blender; layers from an earlier run with the same settings are kept."""
    s = cfg["tissue_movie"]["segments"]
    paths = fetch(s, volume, work)
    rows = plan[layer_rows(plan)]
    out = Path(frames_dir) / "meshes"
    out.mkdir(parents=True, exist_ok=True)
    origin = [float(v) for v in cfg["movie"]["target"]]
    settings = {k: v for k, v in s.items() if k != "blender"} | {"size": list(size), "origin": origin}
    stamp = out / "settings.json"
    text = json.dumps(settings, sort_keys=True)
    if overwrite or not stamp.exists() or stamp.read_text() != text:
        for old in out.glob("m*.png"):
            old.unlink()
        cli.atomic_write(stamp, lambda p: Path(p).write_text(text))
    frames = []
    for row in rows.itertuples():
        path = out / f"m{row.frame:05d}.png"
        if path.exists():
            continue
        cam = fly_camera(s, row.zoom, row.cx, row.cy, row.z, row.reveal, row.mesh_from, focal, up)
        lo, hi = shown(row.z, row.mesh_from)
        frames.append({"path": str(path), "camera": blender_camera(cam, size, focal, origin),
                       "clip": [lo - origin[2], hi - origin[2]]})
    if not frames:
        log.info("mesh layers: all %d done", len(rows))
        return out
    look = ("samples", "roughness", "subsurface", "subsurface_scale", "hdri", "hdri_rotation_deg", "hdri_strength",
            "exposure")
    job = {"size": list(size), "origin": origin, "frames": frames, **{k: s[k] for k in look},
           "meshes": [{"path": str(p), "color": c} for p, c in zip(paths, colors(s))]}
    job_path = out / "job.json"
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
    return out
