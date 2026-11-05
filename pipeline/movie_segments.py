"""Segment meshes rising out of the cut in the tissue movie's fly-through (``tissue_movie.segments``).

At full zoom (in the fly-through's hold, ``movie.hold_end_s``) the scroll eases to a stop; the meshes of
the chosen segments then rise out of the cut towards the camera, as deep as ``depth`` s0 voxels, while the
camera turns (``yaw_deg``, ``tilt_deg``) and pulls back (``pull``) about the point on the cut it looks at;
they hold, sink back into the cut as the camera comes face-on again, and the scroll carries on. What the
meshes show is the segments' part in front of the cut: the tissue that the scroll has milled away.

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
    "start_s": 1.0,       # the reveal starts this long after the fly-through's zoom ends (in movie.hold_end_s)
    "pause_s": 1.0,       # the scroll eases to a stop over this long before it, and back up after it
    "rise_s": 2.5,        # the meshes rise out of the cut as the camera turns and pulls back ...
    "hold_s": 2.0,        # ... stay ...
    "fall_s": 2.0,        # ... and sink back as the camera comes face-on again
    "depth": 1500,        # how far in front of the cut the meshes reach (s0 voxels; 12 um at 8 nm)
    "yaw_deg": 25,        # the camera's turn about the y axis ...
    "tilt_deg": 12,       # ... its rise towards up on screen ...
    "pull": 1.8,          # ... and its distance (x the face-on distance), at the height of the reveal
    "samples": 32,        # Blender (Cycles) samples per px, denoised
    "blender": None,      # the Blender executable; null: $EM_PIPE_BLENDER, else blender
}

# Distinct, saturated colours (sRGB, 0-1), in order.
PALETTE = [(0.90, 0.25, 0.30), (0.20, 0.60, 0.95), (0.98, 0.75, 0.15), (0.35, 0.80, 0.35), (0.75, 0.40, 0.90),
           (0.10, 0.80, 0.80), (0.98, 0.50, 0.20), (0.95, 0.45, 0.75), (0.55, 0.85, 0.15), (0.30, 0.35, 0.95),
           (0.85, 0.85, 0.85), (0.60, 0.35, 0.20)]


def enabled(m):
    return bool(m["segments"]["ids"])


def envelope(t, s, t_end_zoom):
    """The reveal (0-1) at times ``t``: up over rise_s from start_s after the zoom's end, held over hold_s,
    down over fall_s."""
    r0 = t_end_zoom + s["start_s"]
    r1 = r0 + s["rise_s"]
    r2 = r1 + s["hold_s"]
    return ease(t, r0, r1) * (1 - ease(t, r2, r2 + s["fall_s"]))


def add_reveal(fly, mv, s):
    """The fly-through's timeline (pipeline.movie.timeline, no hold at the start) with the reveal: column
    ``reveal`` (0-1) and the scroll paused around it (z held from pause_s before it to pause_s after)."""
    t = fly["t"].to_numpy()
    t1 = mv["zoom_s"]
    r0 = t1 + s["start_s"]
    r3 = r0 + s["rise_s"] + s["hold_s"] + s["fall_s"]
    if min(s["start_s"], s["pause_s"], s["hold_s"]) < 0 or min(s["rise_s"], s["fall_s"]) <= 0:
        raise ValueError("tissue_movie.segments: the times must be >= 0, rise_s and fall_s > 0")
    if r3 + s["pause_s"] > t1 + mv["hold_end_s"] + 1e-9:
        raise ValueError(f"tissue_movie.segments: the reveal and the scroll's pause need movie.hold_end_s >= "
                         f"{s['start_s'] + s['rise_s'] + s['hold_s'] + s['fall_s'] + s['pause_s']:g}")
    z = fly["z"].to_numpy()
    v = np.gradient(z, t) if len(t) > 1 else np.zeros_like(z)
    stop = ease(t, r0 - s["pause_s"], r0) * (1 - ease(t, r3, r3 + s["pause_s"]))   # 1: stopped
    lost = np.concatenate([[0], np.cumsum((stop[1:] * v[1:] + stop[:-1] * v[:-1]) / 2 * np.diff(t))])
    out = fly.copy()
    out["z"] = z - lost
    out["reveal"] = envelope(t, s, t1)
    return out


def view(s, reveal):
    """(yaw, tilt in radians, distance factor, mesh depth in s0 voxels) at ``reveal`` (0-1)."""
    r = float(reveal) if np.isfinite(reveal) else 0.0
    return (math.radians(s["yaw_deg"]) * r, math.radians(s["tilt_deg"]) * r, 1 + (s["pull"] - 1) * r, s["depth"] * r)


def fly_camera(s, zoom, cx, cy, z, reveal, focal, up):
    """The fly-through's camera at a frame: face-on to the cut z + 0.5 at the distance that shows it at
    ``zoom``, turned, raised and pulled back about the point it looks at by the reveal."""
    yaw, tilt, pull, _ = view(s, reveal)
    return camera((cx, cy, z + 0.5), yaw, tilt, focal * zoom * pull, up)


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


def render_layers(cfg, plan, frames_dir, volume, work, size, focal, up, overwrite=False):
    """Render the mesh layers (RGBA PNG, ``frames_dir``/meshes/m<frame>.png) of the frames whose reveal
    is above 0, with Blender; layers from an earlier run with the same settings are kept."""
    s = cfg["tissue_movie"]["segments"]
    paths = fetch(s, volume, work)
    rows = plan[(plan["part"] == "fly") & (plan["reveal"].fillna(0) > 0)] if "reveal" in plan.columns else plan[:0]
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
        cam = fly_camera(s, row.zoom, row.cx, row.cy, row.z, row.reveal, focal, up)
        _, _, _, depth = view(s, row.reveal)
        zc = row.z + 0.5 - origin[2]
        frames.append({"path": str(path), "camera": blender_camera(cam, size, focal, origin),
                       "clip": [zc - depth, zc]})
    if not frames:
        log.info("mesh layers: all %d done", len(rows))
        return out
    job = {"size": list(size), "samples": s["samples"], "origin": origin, "frames": frames,
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
