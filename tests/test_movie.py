"""movie: fly-through frames from an OME-Zarr pyramid, and the MP4."""

import math
import shutil
import warnings

import cv2
import numpy as np
import pytest

from pipeline import movie, omezarr
from pipeline.config import load_config

SIZE = [64, 48]
MOVIE = {"size": SIZE, "fps": 10, "slices_per_s": 2, "hold_start_s": 1.0, "zoom_s": 2.0, "hold_end_s": 1.0,
         "target": [150, 90, 20], "scale_bar": False}


def _volume(root, shape=(40, 160, 256), num_scales=3):
    """Random pyramid (each level the 2x mean of the previous one); returns the levels' data."""
    rng = np.random.default_rng(0)
    data = [rng.integers(1, 255, shape, dtype=np.uint8)]
    shapes = omezarr.create(root, shape, (8.0, 8.0, 8.0), num_scales=num_scales, chunk=(8, 32, 32), shard=(8, 64, 64))
    for s, sh in enumerate(shapes):
        if s:
            prev = data[-1].astype(np.float32)
            pad = [(0, 2 * n - p) for n, p in zip(sh, prev.shape)]
            prev = np.pad(prev, pad, mode="edge")
            data.append(prev.reshape(sh[0], 2, sh[1], 2, sh[2], 2).mean(axis=(1, 3, 5)).round().astype(np.uint8))
        omezarr.open_scale(root, s)[...].write(data[s]).result()
    return data


def _cfg(tmp_path, **m):
    return load_config(_write(tmp_path, **m), step_defaults=movie.DEFAULTS)


def _write(tmp_path, **m):
    import yaml
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump({"name": "vol", "raw_dir": str(tmp_path), "output_dir": str(tmp_path / "out"),
                                    "movie": {**MOVIE, **m}}))
    return path


def test_timeline_zooms_from_the_whole_slice_to_the_target(tmp_path):
    m = _cfg(tmp_path)["movie"]
    plan = movie.timeline(m, (40, 160, 256))
    assert len(plan) == 40 and np.allclose(np.diff(plan["z"]), 0.2)
    first, end = plan.iloc[0], plan.iloc[30]                    # the zoom ends at 4 - 1 = 3 s
    assert first["zoom"] == pytest.approx(max(256 / 64, 160 / 48))   # whole slice fits the frame
    assert (first["cx"], first["cy"]) == pytest.approx((128, 80))
    assert (end["zoom"], end["cx"], end["cy"], end["z"]) == pytest.approx((1, 150, 90, 20))
    assert (plan["zoom"].iloc[:11] == plan["zoom"].iloc[0]).all() and (plan["zoom"].iloc[30:] == 1).all()
    assert (np.diff(plan["zoom"]) <= 1e-12).all()               # only ever zooms in
    with pytest.raises(ValueError, match="target z must be between 6 and 37"):
        movie.timeline({**m, "target": [150, 90, 3]}, (40, 160, 256))
    with pytest.raises(ValueError, match="set movie.target"):
        movie.timeline({**m, "target": None}, (40, 160, 256))


def test_scroll_speed_follows_the_zoom(tmp_path):
    """[zoomed out, zoomed in]: the zoomed-out speed while on the whole slice, easing to the zoomed-in
    speed as the zoom ends, and the target slice on screen when it does."""
    m = {**_cfg(tmp_path)["movie"], "slices_per_s": [6, 2], "target": [150, 90, 15]}
    plan = movie.timeline(m, (40, 160, 256))
    step = np.diff(plan["z"]) * 10                              # slices/s between frames
    assert step[:10] == pytest.approx(6) and step[30:] == pytest.approx(2)
    assert (np.diff(step) <= 1e-9).all()                        # only ever slows down
    assert plan["z"].iloc[30] == pytest.approx(15)
    assert plan["z"].iloc[0] == pytest.approx(15 - 6 - (6 + 2) / 2 * 2)    # 1 s at 6, 2 s averaging 4
    with pytest.raises(ValueError, match="target z must be between 14 and 37"):
        movie.timeline({**m, "target": [150, 90, 10]}, (40, 160, 256))


def test_frames_come_from_the_right_level(tmp_path):
    data = _volume(tmp_path / "v.ome.zarr")
    levels = movie.open_levels(tmp_path / "v.ome.zarr", 0.1)
    # zoom 1, view centre on whole voxels: exactly the s0 crop
    frame, level = movie.render_frame(levels, SIZE, 1.0, 100, 60, 7)
    assert level == 0 and np.array_equal(frame, data[0][7, 36:84, 68:132])
    # zoom 4 reads s2 (40 x 64 px; this view starts at its corner and runs 8 rows past its end);
    # slice 7 lies between s2 planes 1 and 2 (their centres: s0 z 5.5 and 9.5)
    frame, level = movie.render_frame(levels, SIZE, 4.0, 128, 96, 7)
    assert level == 2
    want = 0.625 * data[2][1].astype(float) + 0.375 * data[2][2]
    assert np.abs(frame[:40].astype(float) - want).max() <= 0.5 and not frame[40:].any()
    # outside the volume: black
    frame, _ = movie.render_frame(levels, SIZE, 1.0, -100, -100, 0)
    assert not frame.any()


def test_frames_and_encode(tmp_path, caplog):
    _volume(tmp_path / "out" / "vol.ome.zarr")
    cfg = load_config(_write(tmp_path, scale_bar=True), step_defaults=movie.DEFAULTS)
    assert movie.main(["--config", str(cfg["config_path"]), "frames", "--frames", "0,30"]) == 0
    out = tmp_path / "out" / "work" / "movie" / "vol_movie"
    assert sorted(p.name for p in out.glob("f*.png")) == ["f00000.png", "f00030.png"]
    first = cv2.imread(str(out / "f00000.png"), cv2.IMREAD_UNCHANGED)
    assert first.shape == (48, 64) and first.dtype == np.uint8
    with pytest.raises(RuntimeError, match="38 of 40 frames missing"):
        movie.encode(cfg)
    assert movie.main(["--config", str(cfg["config_path"]), "frames"]) == 0
    assert len(list(out.glob("f*.png"))) == 40
    stamp = (out / "f00000.png").stat().st_mtime_ns
    assert movie.main(["--config", str(cfg["config_path"]), "frames"]) == 0      # resumes: nothing redone
    assert (out / "f00000.png").stat().st_mtime_ns == stamp
    cmd = movie.encode_command(cfg)
    assert cmd[-1] == str(tmp_path / "out" / "vol_movie.mp4") and "libx264" in cmd
    # a changed timeline starts afresh
    cfg2 = _write(tmp_path, scale_bar=True, target=[100, 90, 20])
    assert movie.main(["--config", str(cfg2), "frames", "--frames", "5"]) == 0
    assert [p.name for p in out.glob("f*.png")] == ["f00005.png"]
    if shutil.which("ffmpeg"):
        assert movie.main(["--config", str(cfg2), "frames"]) == 0
        assert movie.main(["--config", str(cfg2), "encode"]) == 0
        assert (tmp_path / "out" / "vol_movie.mp4").stat().st_size > 1000


def test_scale_bar_lengths():
    assert [movie.nice_length(v) for v in (0.9, 1, 3.3, 7, 19.9, 20, 4500)] == [0.5, 1, 2, 5, 10, 20, 2000]
    frame = movie.draw_scale_bar(np.full((1080, 1920), 128, np.uint8), 8.0)     # 8 nm/px: 1 um bar = 125 px
    rows = np.where((frame == 255).sum(axis=1) >= 120)[0]
    assert len(rows) and (frame[rows[0]] == 255).sum() in range(124, 128)


def test_turned_plane(tmp_path):
    """Turned 90 degrees the frame is the z-y plane (frame columns along z); slightly turned, it
    matches the x-y frame."""
    data = _volume(tmp_path / "v.ome.zarr", shape=(96, 64, 80))
    levels = movie.open_levels(tmp_path / "v.ome.zarr", 0.1)
    frame, level = movie.render_frame(levels, SIZE, 1.0, 40.5, 32, 47.5, rotation=90)
    # column u: z = 47.5 + (u + 0.5 - 32) -> slice 16 + u; row v: y = 8 + v; x = 40
    assert level == 0 and np.array_equal(frame, data[0][16:80, 8:56, 40].T)
    flat, _ = movie.render_frame(levels, SIZE, 1.0, 40, 32, 47)
    near, _ = movie.render_frame(levels, SIZE, 1.0, 40, 32, 47, rotation=0.001)
    assert np.abs(near.astype(int) - flat).max() <= 1


def test_timeline_with_rotation(tmp_path):
    m = {**_cfg(tmp_path)["movie"], "rotate_deg": 90, "rotate_s": 1.0, "rotated_s": 1.0, "slices_per_s": 2,
         "target": [150, 90, 50]}                               # the default turn: the plane, face-on
    plan = movie.timeline(m, (200, 160, 256))
    assert len(plan) == 60 and (plan["rotation"].iloc[:41] == 0).all() and plan["rotation"].iloc[50:].eq(90).all()
    # turned: the view moves along -x instead of z (the plane's normal), at the same 2 voxels/s
    tail = plan.iloc[50:]
    assert np.diff(tail["cx"]) * 10 == pytest.approx(-2) and np.diff(tail["z"]) == pytest.approx(0, abs=1e-9)
    assert (plan["cx"].iloc[:31] - plan["cx"].iloc[30]).abs().max() > 0 and plan["cx"].iloc[30] == pytest.approx(150)
    # the turned frame spans 32 slices either side of its centre, ~3 slices past the target by then
    with pytest.raises(ValueError, match="from -29 to \\+35 slices .* between 29 and 163"):
        movie.timeline({**m, "target": [150, 90, 20]}, (200, 160, 256))


def test_view_rotation_turns_the_picture_not_mirrors_it(tmp_path):
    _volume(tmp_path / "out" / "vol.ome.zarr")
    for deg in (0, 180):
        movie.main(["--config", str(_write(tmp_path, view_rotation_deg=deg)), "frames", "--frames", "35"])
        out = tmp_path / "out" / "work" / "movie" / "vol_movie" / "f00035.png"
        frame = cv2.imread(str(out), cv2.IMREAD_UNCHANGED)
        if deg == 0:
            upright = frame
    assert np.array_equal(frame, np.rot90(upright, 2)) and not np.array_equal(frame, upright[::-1])
    with pytest.raises(ValueError, match="0 or 180"):
        movie.timeline({**_cfg(tmp_path)["movie"], "view_rotation_deg": 90}, (40, 160, 256))


def test_side_view_is_the_y_z_plane(tmp_path):
    """At 90 degrees the frame is the y-z plane (columns along z) from the right level, and it joins
    the turning sampler without a jump."""
    data = _volume(tmp_path / "v.ome.zarr", shape=(96, 64, 80))
    levels = movie.open_levels(tmp_path / "v.ome.zarr", 0.1)
    frame, level = movie.render_frame(levels, SIZE, 1.0, 40.5, 32, 47.5, rotation=90)
    assert level == 0 and np.array_equal(frame, data[0][16:80, 8:56, 40].T)
    near, _ = movie.render_frame(levels, SIZE, 1.0, 40.5, 32, 47.5, rotation=89.9999)
    assert np.abs(near.astype(int) - frame).max() <= 1
    back, _ = movie.render_frame(levels, SIZE, 1.0, 40.5, 32, 47.5, rotation=-90)
    np.testing.assert_array_equal(back, frame[:, ::-1])
    # zoomed out 2x: level 1, the x plane between two s1 planes blended
    frame, level = movie.render_frame(levels, SIZE, 2.0, 40.0, 32, 47.5, rotation=90)
    assert level == 1 and frame.shape == (48, 64)


def test_timeline_zooms_out_to_the_side_view(tmp_path):
    m = {**_cfg(tmp_path)["movie"], "rotate_deg": 90, "rotate_s": 1.0, "rotated_s": 0.5, "zoom_out_s": 1.0,
         "hold_out_s": 0.5, "slices_per_s": [6, 2], "target": [150, 90, 50], "rotate_style": "plane"}
    plan = movie.timeline(m, (300, 160, 256))
    assert len(plan) == 70                                     # 1 + 2 + 1 + 1 + 0.5 + 1 + 0.5 s at 10 fps
    end = plan.iloc[-1]
    assert end["zoom"] == pytest.approx(max(300 / 64, 160 / 48)) and end["rotation"] == 90
    assert (end["cy"], end["z"]) == pytest.approx((80, 149.5))  # the centre of the y-z plane
    held = plan.iloc[50:55]                                    # rotated hold: full resolution, turned
    assert (held["zoom"] == 1).all() and (held["rotation"] == 90).all()
    step = -np.diff(plan["cx"].iloc[50:]) * 10                 # turned: the scroll runs along -x
    assert step[0] == pytest.approx(2) and step[-1] == pytest.approx(6)   # and speeds back up zoomed out
    with pytest.raises(ValueError, match="rotate_deg 0, 90 or -90"):
        movie.timeline({**m, "rotate_deg": 45}, (300, 160, 256))


def test_x_z_plane(tmp_path):
    data = _volume(tmp_path / "v.ome.zarr", shape=(96, 64, 80))
    levels = movie.open_levels(tmp_path / "v.ome.zarr", 0.1)
    # column u: x = 40 + (u + 0.5 - 32) -> voxel 8 + u; row v: slice 16 + v; y = 20.5 -> row 20
    frame, level = movie._plane_frame(levels, (64, 64), 1.0, 40, 48, 20.5, "y")
    assert level == 0 and np.array_equal(frame, data[0][16:80, 20, 8:72])


def test_timeline_block_turn(tmp_path):
    """A block turn: the scroll eases to a stop before it and back up after it; turned, the view is
    the block's side face, half a frame to the side and half a frame width deeper."""
    m = {**_cfg(tmp_path)["movie"], "rotate_deg": 90, "rotate_s": 1.0, "rotated_s": 1.0, "rotate_pause_s": 0.5,
         "slices_per_s": 2, "target": [150, 90, 50], "rotate_style": "block"}
    plan = movie.timeline(m, (200, 160, 256))
    assert len(plan) == 60
    turning = plan[(plan["turn"] > 0) & (plan["turn"] < 1)]
    assert list(turning.index) == list(range(41, 50))
    still = plan.iloc[40:50]                                   # stopped from the turn's start (4 s) to its end
    assert (still["cx"] == still["cx"].iloc[0]).all() and (still["z"] == still["z"].iloc[0]).all()
    assert np.diff(plan["z"].iloc[:36]) == pytest.approx(0.2)  # full speed until the pause (3.5 s)
    front, turned = plan.iloc[49], plan.iloc[50]
    assert turned["turn"] == 1 and turned["rotation"] == pytest.approx(90)
    assert (turned["cx"] - front["cx"], turned["z"] - front["z"]) == pytest.approx((32, 32))
    step = -np.diff(plan["cx"].iloc[50:]) * 10                 # then scrolls along -x, speeding back up
    assert step[0] == pytest.approx(0, abs=0.1) and step[-1] == pytest.approx(2)
    assert (np.diff(plan["z"].iloc[50:]) == 0).all()
    with pytest.raises(ValueError, match="block needs rotate_deg 90 or -90"):
        movie.timeline({**m, "rotate_deg": 45}, (200, 160, 256))
    with pytest.raises(ValueError, match="outside the volume"):          # the block reaches 64 slices deep
        movie.timeline(m, (100, 160, 256))
    for bad in ({"rotate_pause_s": -1}, {"rotate_tilt_deg": 90}, {"rotate_fov_deg": 170}, {"rotate_pull_back": 0.3}):
        with pytest.raises(ValueError, match="rotate_pause_s must be|cut into the block"):
            movie.timeline({**m, **bad}, (200, 160, 256))
    # the turn ends between two frames and the zoom-out starts at once: it still ends on the side's centre
    end = movie.timeline({**m, "rotate_s": 1.05, "rotated_s": 0.0, "zoom_out_s": 1.0, "hold_out_s": 0.5},
                         (200, 160, 256)).iloc[-1]
    assert (end["cy"], end["z"]) == pytest.approx((80, 99.5))


def test_block_turn_starts_and_ends_on_the_flat_frames(tmp_path):
    _volume(tmp_path / "v.ome.zarr", shape=(96, 64, 80))
    levels = movie.open_levels(tmp_path / "v.ome.zarr", 0.1)
    for side, up in ((1, -1), (-1, 1)):
        block = movie.block_faces(levels, SIZE, 1.0, 40, 32, 16, side, up)
        flat, _ = movie.render_frame(levels, SIZE, 1.0, 40, 32, 16)
        turned, _ = movie.render_frame(levels, SIZE, 1.0, 40 + 32 * side, 32, 48, 90 * side)
        start = movie.block_view(block, SIZE, 1e-6, 90 * side, 40, 20, None, up)
        end = movie.block_view(block, SIZE, 1 - 1e-6, 90 * side, 40, 20, None, up)
        assert np.abs(start.astype(int) - flat).max() <= 1 and np.abs(end.astype(int) - turned).max() <= 1
        mid = movie.block_view(block, SIZE, 0.5, 90 * side, 40, 20, None, up)
        assert (mid[[0, 0, -1, -1], [0, -1, 0, -1]] == 0).all() and mid[24, 32] > 0   # whole block in frame
        assert block["pull_back"] > 1
        # faces of one grey value: shown as they are (no shading), the top one (200) on the side up on screen
        block["faces"] = [(np.full(f[0].shape, v, np.float32), *f[1:]) for f, v in zip(block["faces"], (50, 100, 200))]
        mid = movie.block_view(block, SIZE, 0.5, 90 * side, 40, 20, None, up)
        inside = cv2.erode((mid > 0).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        assert mid[inside].min() >= 50 and (mid == 100).any()           # edges between faces blend them
        rows = np.nonzero(mid == 200)[0]
        assert len(rows) and (rows.max() < 24 if up == -1 else rows.min() > 24)
    # zoomed, w / 2 odd
    block = movie.block_faces(levels, [66, 50], 1.25, 40, 32, 10, 1, 1)
    turned, _ = movie.render_frame(levels, [66, 50], 1.25, 40 + 41.25, 32, 10 + 41.25, 90)
    end = movie.block_view(block, [66, 50], 1 - 1e-6, 90, 40, 20, None, 1)
    assert np.abs(end.astype(int) - turned).max() <= 1


def test_block_turn_frames(tmp_path):
    """Mid-turn frames show a solid block, also when the camera dips below it (a negative tilt)."""
    _volume(tmp_path / "out" / "vol.ome.zarr", shape=(160, 160, 256))
    out = tmp_path / "out" / "work" / "movie" / "vol_movie"
    for tilt in (20, -20):
        cfg = _write(tmp_path, rotate_deg=-90, rotate_s=1.0, rotated_s=0.5, rotate_pause_s=0.5, view_rotation_deg=180,
                     target=[150, 90, 50], rotate_tilt_deg=tilt, rotate_style="block")
        movie.main(["--config", str(cfg), "frames", "--frames", "38,45,52"])
        mid = cv2.imread(str(out / "f00045.png"), cv2.IMREAD_UNCHANGED)
        assert mid.shape == (48, 64) and mid[0, 0] == 0 and mid[24, 32] > 0
        hull = np.zeros_like(mid)
        cv2.fillConvexPoly(hull, cv2.convexHull(np.argwhere(mid > 0)[:, ::-1].astype(np.int32)), 1)
        assert ((hull > 0) & (mid == 0)).sum() < 0.08 * hull.sum()        # no face missing
        assert (out / "f00038.png").exists() and (out / "f00052.png").exists()


def test_render_faces_antialiased_without_seams():
    """Faces sharing an edge leave no seam, the silhouette covers its area (to the coverage samples),
    and later faces cover earlier ones."""
    size, focal = (160, 120), 80 / math.tan(math.radians(20))
    cam = movie.camera((0, 0, 0), math.radians(35), math.radians(20), 400, -1)
    flat = np.full((40, 40), 100, np.float32)
    faces = [(flat, (-50, -50, -50), (100, 0, 0), (0, 100, 0), (0, 0, -1)),     # a cube's front, +x side and top
             (flat, (50, -50, -50), (0, 0, 100), (0, 100, 0), (1, 0, 0)),
             (flat, (-50, -50, -50), (100, 0, 0), (0, 0, 100), (0, -1, 0))]
    out = movie.render_faces(faces, size, cam, focal, shade=0)
    inside = cv2.erode((out > 0).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    assert out[inside] == pytest.approx(100, abs=1e-3)
    corners = np.array([(x, y, z) for x in (-50, 50) for y in (-50, 50) for z in (-50, 50)], float)
    hull = cv2.convexHull(movie.project(corners, cam, size, focal).astype(np.float32))
    assert out.sum() / 100 == pytest.approx(cv2.contourArea(hull), abs=2)
    over = movie.render_faces([(flat / 2, *faces[0][1:]), faces[0]], size, cam, focal, shade=0)
    np.testing.assert_allclose(over, movie.render_faces(faces[:1], size, cam, focal, shade=0), atol=1e-3)
    # coverage on request: what faces of ones would show; the picture is the same
    tex = [(np.random.default_rng(0).uniform(1, 255, (40, 40)).astype(np.float32), *f[1:]) for f in faces]
    pic, cover = movie.render_faces(tex, size, cam, focal, coverage=True)
    np.testing.assert_array_equal(pic, movie.render_faces(tex, size, cam, focal))
    np.testing.assert_allclose(cover, out / 100, atol=1e-6)
    assert cover.max() == pytest.approx(1) and cover.min() == 0


def test_render_faces_edge_on():
    """A face seen (nearly) edge-on: no division warnings, nothing but finite grey values."""
    size, focal = (64, 48), 32 / math.tan(math.radians(20))
    img = np.random.default_rng(0).uniform(0, 255, (48, 64)).astype(np.float32)
    face = (img, (-32, -24, 0), (64, 0, 0), (0, 48, 0), (0, 0, -1))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for yaw in (89.9999999, 90 - 1e-12, 90 - 1e-6):
            out = movie.render_faces([face], size, movie.camera((0, 0, 0), math.radians(yaw), 0, 300, -1), focal)
            assert np.isfinite(out).all() and out.max() <= 255


def test_public_names():
    assert (movie.plane_frame, movie.pyramid_level, movie.ease, movie.integral) == \
        (movie._plane_frame, movie._level, movie._ease, movie._integral)


def test_antialiasing_follows_the_shrink():
    """A face receding in perspective is blurred where the view shrinks it, and not where it does not;
    just off face-on (a turn starting) it keeps its detail."""
    rng = np.random.default_rng(0)
    img = rng.normal(128, 40, (64, 256)).astype(np.float32)
    src = np.float32([[-0.5, -0.5], [255.5, -0.5], [255.5, 63.5], [-0.5, 63.5]])
    m = cv2.getPerspectiveTransform(src, np.float32([[0, 0], [60, 28], [60, 36], [0, 64]]) - 0.5)   # 1x to ~8x
    out = movie._antialiased(img, m)
    assert out[:, :16].std() > 0.9 * img.std() and out[:, -16:].std() < 0.1 * img.std()
    img = rng.uniform(0, 255, (180, 320)).astype(np.float32)
    block = {"faces": [(img, (-160, -90, 0.5), (320, 0, 0), (0, 180, 0), (0, 0, -1))], "centre": (0, 0, 160.5),
             "half": (160, 90, 160)}
    view = movie.block_view(block, (320, 180), 1e-3, 90, 40, 20, 1.8, -1).astype(np.float32)
    detail = [np.abs(cv2.Laplacian(a, cv2.CV_32F))[20:-20, 20:-20].mean() for a in (view, img)]
    assert detail[0] > 0.75 * detail[1]                              # bilinear resampling: ~0.55
    # and without ringing: a step from 100 to 200 stays between them (unclamped Lanczos: ~90-210)
    block["faces"] = [(np.where(np.arange(320) < 161, 100, 200).astype(np.float32)[None].repeat(180, 0),
                       *block["faces"][0][1:])]
    view = movie.block_view(block, (320, 180), 1e-3, 90, 40, 20, 1.8, -1)[5:-5, 5:-5]
    assert view.min() >= 100 and view.max() <= 200 and len(np.unique(view)) > 2


def test_timeline_turns_during_the_zoom_out(tmp_path):
    """rotate_start_zoom: the zoom-out starts face-on and the plane turns from when it reaches that zoom,
    ending on the side view's centre; the turned frames may reach out of the volume (black: no data)."""
    m = {**_cfg(tmp_path)["movie"], "rotate_deg": 90, "rotate_s": 0.5, "rotated_s": 1.0, "zoom_out_s": 2.0,
         "hold_out_s": 0.5, "rotate_start_zoom": 2.0, "target": [150, 90, 50]}
    plan = movie.timeline(m, (300, 160, 256))
    assert len(plan) == 65                                     # 1 + 2 + 1 + 2 + 0.5 s: no turn or turned hold
    turning = plan[(plan["rotation"] > 0) & (plan["rotation"] < 90)]
    assert (plan["zoom"].iloc[30:41] == 1).all() and (plan["rotation"].iloc[:41] == 0).all()   # held, face-on
    assert plan["zoom"].iloc[turning.index[0] - 1] <= 2 <= turning["zoom"].iloc[0]
    end = plan.iloc[-1]
    assert end["rotation"] == 90 and turning.index[-1] < 60 and (end["cy"], end["z"]) == pytest.approx((80, 149.5))
    assert np.abs(np.diff(plan["z"])).max() < 10 and np.abs(np.diff(plan["cx"])).max() < 10      # no jumps
    with pytest.raises(ValueError, match="would end 0.43 s after"):
        movie.timeline({**m, "rotate_s": 1.5}, (300, 160, 256))
    for bad in ({"rotate_style": "block"}, {"rotate_start_zoom": 5.0}, {"rotate_deg": 45}):
        with pytest.raises(ValueError):
            movie.timeline({**m, **bad}, (300, 160, 256))


def test_turned_frames_antialiased(tmp_path):
    """Zoomed out, the turned plane is blurred as _plane_frame blurs: no jump as the turn starts or ends."""
    _volume(tmp_path / "v.ome.zarr", shape=(160, 160, 256))
    levels = movie.open_levels(tmp_path / "v.ome.zarr", 0.1)
    for zoom in (1.5, 3.9):                                   # level 0 and 1, 1.5-1.95 level px per frame px
        for rot in (0, 90):
            plane, _ = movie.render_frame(levels, SIZE, zoom, 128, 80, 80, rotation=rot)
            near, _ = movie.render_frame(levels, SIZE, zoom, 128, 80, 80, rotation=abs(rot - 1e-4))
            assert np.abs(near.astype(int) - plane).max() <= 1          # unblurred: up to ~50
