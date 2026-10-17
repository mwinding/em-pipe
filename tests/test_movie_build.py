"""movie_build: the stack built up slice by slice in 3D, handing over to the fly-through."""

import math
import shutil

import cv2
import numpy as np
import pytest
import yaml

from pipeline import movie, omezarr
from pipeline import movie_build as mb
from pipeline.config import deep_merge, load_config

SHAPE = (120, 80, 128)
MOVIE = {"size": [64, 48], "fps": 10, "slices_per_s": 2, "hold_start_s": 1.0, "zoom_s": 2.0, "hold_end_s": 1.0,
         "target": [64, 40, 30], "view_rotation_deg": 180}
BUILD = {"lead_in_s": 0.2, "start_rate": 4.0, "slow_s": 0.5, "ramp_s": 1.0, "max_rate": 100, "stop_s": 0.4,
         "orbit_delay_s": 0.1, "orbit_s": 0.8, "fade_s": 0.3, "scroll_ease_s": 0.5}
SIZE = MOVIE["size"]
FOCAL = 32 / math.tan(math.radians(50) / 2)


def _volume(root):
    """Random tissue in a footprint that shrinks at slice 80 (zero outside it; one slice with a corner
    missing), as a 3-level pyramid of 2x means."""
    rng = np.random.default_rng(0)
    data = rng.integers(1, 255, SHAPE, dtype=np.uint8)
    mask = np.zeros(SHAPE, bool)
    mask[:80, 6:74, 8:120] = True
    mask[80:, 16:64, 24:104] = True
    mask[50, 6:30, 8:40] = False
    data[~mask] = 0
    shapes = omezarr.create(root, SHAPE, (8.0, 8.0, 8.0), num_scales=3, chunk=(8, 32, 32), shard=(8, 64, 64))
    for s, sh in enumerate(shapes):
        if s:
            prev = np.pad(data.astype(np.float32), [(0, 2 * n - p) for n, p in zip(sh, data.shape)], mode="edge")
            data = prev.reshape(sh[0], 2, sh[1], 2, sh[2], 2).mean(axis=(1, 3, 5)).round().astype(np.uint8)
        omezarr.open_scale(root, s)[...].write(data).result()


def _write(tmp_path, **build):
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump({"name": "vol", "raw_dir": str(tmp_path), "output_dir": str(tmp_path / "out"),
                                    "movie": MOVIE, "build": {**BUILD, **build}}))
    return path


def _setup(tmp_path, **build):
    _volume(tmp_path / "out" / "vol.ome.zarr")
    cfg = load_config(_write(tmp_path, **build), step_defaults=deep_merge(movie.DEFAULTS, mb.DEFAULTS))
    return cfg, movie.open_levels(tmp_path / "out" / "vol.ome.zarr", 0.1)


def test_footprints(tmp_path):
    """One box per footprint, one voxel inside the tissue; the slice with a corner missing changes
    nothing, and the changeover slice goes with the smaller footprint."""
    _, levels = _setup(tmp_path)
    assert mb.footprints(levels, 24, 120) == [mb.Box(24, 79, 9, 119, 7, 73), mb.Box(79, 120, 25, 103, 17, 63)]
    assert mb.footprints(levels, 24, 120, min_depth=0.5) == [mb.Box(24, 120, 25, 103, 17, 63)]   # too thin: joined


def test_imaging_ramps_up_and_stops_on_the_last_slice():
    b = {**mb.DEFAULTS["build"], "lead_in_s": 0.5, "start_rate": 2.0, "slow_s": 1.0, "ramp_s": 3.0, "max_rate": 500,
         "stop_s": 1.0}
    t, n, rate, t_stop = mb.imaging(b, 2000)
    assert not n[t < 0.5].any() and n[t >= 0.5][0] == 1                                 # the first slice at 0.5 s
    assert np.interp(1.5, t, n) == pytest.approx(3, abs=1e-2)                            # then 2/s for 1 s
    assert (np.diff(n) >= 0).all() and n[-1] == pytest.approx(2000) and rate[-1] == pytest.approx(0, abs=1e-9)
    assert rate.max() == pytest.approx(500, rel=1e-3) and t_stop > 4.5                     # held at 500/s
    assert t[-1] == pytest.approx(t_stop + 1)
    t, n, rate, t_stop = mb.imaging(b, 150)                                                # too thin for 500/s
    assert rate.max() < 400 and t_stop == pytest.approx(4.5) and n[-1] == pytest.approx(150)
    with pytest.raises(ValueError, match="too few"):
        mb.imaging(b, 6)                                                                  # 1 + 2/s for 5 s


def test_fly_plan_eases_the_scroll_in(tmp_path):
    cfg, _ = _setup(tmp_path)
    m, b = cfg["movie"], cfg["build"]
    plain, eased = movie.timeline(m, SHAPE), mb.fly_plan(m, b, SHAPE)
    assert eased["z"].iloc[0] == pytest.approx(plain["z"].iloc[0] + 2 * 0.5 / 2)          # half the ease's slices
    assert np.allclose(eased["z"].iloc[5:], plain["z"].iloc[5:])
    step = np.diff(eased["z"].iloc[:7])
    assert step[0] < 0.05 and (np.diff(step) > 0).all() and step[-1] == pytest.approx(0.2)
    with pytest.raises(ValueError, match="scroll_ease_s"):
        mb.fly_plan(m, {**b, "scroll_ease_s": 2.0}, SHAPE)


def test_timeline(tmp_path):
    """Whole slices in acquisition order (+z) from the fly-through's first slice; the camera starts
    near the newest slice, from +z, and orbits to face the first slice; the fly-through follows."""
    cfg, levels = _setup(tmp_path)
    fly, stack, end = mb.load(cfg, levels)
    assert fly["z"].iloc[0] == 24.5 and end == 24.5 + math.floor(79 - stack.pad - 24.5)   # short of the next footprint
    assert stack.boxes[0][2:] == (10, 118, 8, 72)              # on the first frame's pixel grid (zoom 2)
    plan = mb.timeline(cfg["movie"], cfg["build"], fly, stack, end)
    build, rest = plan[plan["part"] == "build"], plan[plan["part"] == "fly"]
    assert list(rest["frame"]) == list(range(len(build), len(plan))) and len(rest) == 40
    assert rest["t"].iloc[0] == pytest.approx(len(build) / 10)
    slices = build["slices"].to_numpy()
    assert slices[0] == 0 and slices[2] == 1 and slices[-1] == end - 24.5 and (np.diff(slices) >= 0).all()
    assert (slices == np.round(slices)).all() and (build["lo"] == 24.5).all()
    assert np.array_equal(build["hi"], 24.5 + slices)
    start, last = build.iloc[0], build.iloc[-1]
    assert start["yaw"] == 215 and start["dist"] == pytest.approx(FOCAL * 1.5)              # close, from +z
    assert abs(last["yaw"] - 360) < 8 and abs(last["tilt"]) < 3          # a frame before face-on and last["rim"] == pytest.approx(0.75)
    with pytest.raises(ValueError, match="orbit must end after"):
        mb.timeline(cfg["movie"], {**cfg["build"], "orbit_delay_s": 0, "orbit_s": 0.2}, fly, stack, end)


def test_hand_over_is_the_fly_throughs_first_frame(tmp_path):
    cfg, levels = _setup(tmp_path)
    m, b = cfg["movie"], cfg["build"]
    fly, stack, end = mb.load(cfg, levels)
    plan = mb.timeline(m, b, fly, stack, end)
    f0 = fly.iloc[0]
    rim = mb.rim_face(levels, SIZE, f0.zoom, f0.cx, f0.cy, f0.z)
    want = movie.render_frame(levels, SIZE, f0.zoom, f0.cx, f0.cy, f0.z)[0]
    last = next(plan[plan["part"] == "build"].iloc[[-1]].itertuples())
    face_on = last._replace(lx=f0.cx, ly=f0.cy, lz=f0.z, yaw=360.0, tilt=0.0, dist=FOCAL * f0.zoom, rim=1.0)
    frame = mb.render_build(stack, face_on, SIZE, FOCAL, 1, rim)
    assert np.abs(frame.astype(int) - want).max() <= 1
    off = [np.abs(mb.render_build(stack, row, SIZE, FOCAL, 1, rim).astype(int) - want).mean()
           for row in plan[plan["part"] == "build"].iloc[[-6, -1]].itertuples()]
    assert off[1] < off[0] / 2                                 # and the orbit comes to it smoothly


def test_stack_faces_from_the_newest_slice(tmp_path):
    _, levels = _setup(tmp_path)
    stack = mb.Stack(levels, mb.footprints(levels, 20, 120), 2.0, (0, -8))
    cam = movie.camera((64, 40, 100), math.radians(215), math.radians(20), 300, 1)       # from +z, -x and +y
    faces = stack.faces(20, 100, cam, SIZE, FOCAL)
    assert [tuple(f[4]) for f in faces] == [(-1, 0, 0), (0, 1, 0), (0, 0, 1)] * 2          # the far box first
    tex, corner, du, dv, _ = faces[3]                                                   # the near box's side
    assert tuple(corner) == (26, 18, 79) and tuple(du) == (0, 0, 21) and tuple(dv) == (0, 44, 0)
    assert tex.shape == (22, 11)                                                        # 2 voxels a texel
    tex, corner, du, dv, _ = faces[5]                                                   # the newest slice, 99
    want = movie._plane_frame(levels, (38, 22), 2.0, 64, 40, 99.5, "z")[0]
    assert tuple(corner) == (26, 18, 100) and np.array_equal(tex, want)


def test_stack_textures_follow_the_view(tmp_path):
    """Close up, a face gets finer texels, only for the part in view: face-on at 1 voxel a pixel, the
    stack's end is the s0 frame."""
    _, levels = _setup(tmp_path)
    stack = mb.Stack(levels, mb.footprints(levels, 20, 120), 2.0, (0, -8))
    cam = movie.camera((60, 30, 30), 0.0, 0.0, FOCAL, 1)
    (tex, corner, du, dv, normal), = stack.faces(30, 60, cam, SIZE, FOCAL)
    assert tuple(normal) == (0, 0, -1) and tex.shape == (50, 72)                        # 48 x 64 + 4 a side, from
    assert tuple(corner) == (24, 8, 30) and tuple(du) == (72, 0, 0)                     # the block's edge at y 8
    frame = np.clip(np.rint(movie.render_faces([(tex, corner, du, dv, normal)], SIZE, cam, FOCAL)), 0, 255)
    assert np.abs(frame - movie.render_frame(levels, SIZE, 1.0, 60, 30, 30)[0])[2:].max() <= 1
    assert not frame[:2].any()                                                          # y 6-8: outside the block


def test_clip_keeps_the_part_in_view():
    cam = movie.camera((0, 0, 0), 0.0, 0.0, 100.0, 1)
    square = np.array([[-500, -500, 0], [500, -500, 0], [500, 500, 0], [-500, 500, 0]], float)
    seen = mb._clip(square, cam, (64, 48), 100.0)
    assert np.allclose(np.ptp(seen, 0)[:2], (64, 48))                                  # 1 voxel a pixel at 100
    assert not len(mb._clip(square + (0, 0, -200), cam, (64, 48), 100.0))                # behind the camera


def test_crop_is_exact_on_whole_texels():
    tex = np.arange(40, dtype=np.float32).reshape(2, 20)
    assert np.allclose(mb._crop(tex, 1, 3, 5), tex[:, 3:8]) and np.allclose(mb._crop(tex, 1, 3.5, 5), tex[:, 3:8] + 0.5)
    assert np.allclose(mb._crop(tex.T, 0, 3.5, 5), tex.T[3:8] + 0.5)
    assert mb._crop(tex, 1, 2, 0.3).shape == (2, 1)


def test_frames_and_encode(tmp_path):
    cfg, _ = _setup(tmp_path)
    path = cfg["config_path"]
    assert mb.main(["--config", path, "frames", "--frames", "0,15"]) == 0
    out = tmp_path / "out" / "work" / "movie" / "vol_movie"
    first = cv2.imread(str(out / "f00000.png"), cv2.IMREAD_UNCHANGED)
    assert first.shape == (48, 64) and not first.any()          # black before the first slice
    assert cv2.imread(str(out / "f00015.png"), cv2.IMREAD_UNCHANGED).any()
    with pytest.raises(RuntimeError, match="frames missing"):
        mb.encode(cfg)
    assert mb.main(["--config", path, "frames"]) == 0
    n = len(list(out.glob("f*.png")))
    stamp = (out / "f00000.png").stat().st_mtime_ns
    assert mb.main(["--config", path, "frames"]) == 0             # resumes: nothing redone
    assert (out / "f00000.png").stat().st_mtime_ns == stamp
    # the fly-through's frames are pipeline.movie's
    levels = movie.open_levels(tmp_path / "out" / "vol.ome.zarr", 0.1)
    plan = mb.timeline(cfg["movie"], cfg["build"], *mb.load(cfg, levels))
    assert n == len(plan)
    row = plan[plan["part"] == "fly"].iloc[10]
    want = np.rot90(movie.render_frame(levels, SIZE, row.zoom, row.cx, row.cy, row.z)[0], 2)
    assert np.array_equal(cv2.imread(str(out / f"f{int(row.frame):05d}.png"), cv2.IMREAD_UNCHANGED), want)
    # a changed timeline starts afresh
    path = str(_write(tmp_path, ramp_s=1.5))
    assert mb.main(["--config", path, "frames", "--frames", "5"]) == 0
    assert [p.name for p in out.glob("f*.png")] == ["f00005.png"]
    if shutil.which("ffmpeg"):
        assert mb.main(["--config", path, "frames"]) == 0
        assert mb.main(["--config", path, "encode"]) == 0
        assert (tmp_path / "out" / "vol_movie.mp4").stat().st_size > 1000
