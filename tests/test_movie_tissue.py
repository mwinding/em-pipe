"""movie_tissue: the 3D tissue mask, the slice-stack volume renderer (grey values unchanged, opacity
only), the timeline, the fly-through ending and the CLI."""

import math
import shutil
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml

from pipeline import movie, movie_tissue as mt, omezarr
from pipeline.config import deep_merge, load_config

SIZE = [64, 48]
SHORT = {"size": SIZE, "fps": 10, "block_s": 0.5, "fade_s": 1.0, "tissue_s": 0.5, "ease_s": 0.3, "level": 0,
         "approach_s": 0.5, "handover_s": 0.2, "scroll_ease_s": 0.5, "build_s": 0.0, "turn_s": 0.5, "blend_s": 0.0}


def _tissue_volume(shape=(24, 64, 72), seed=0):
    """(volume, truth): textured tissue in flat resin that has curtaining stripes (along y) and
    slice-to-slice brightness steps, no data (0) in a corner. In the tissue: a large resin pocket (a
    hole), a small one, a flat pale cell (darker than halfway to resin) and a 2-voxel flat speck; in
    the resin, a 2-voxel textured speck."""
    rng = np.random.default_rng(seed)
    nz, ny, nx = shape
    z, y, x = np.ogrid[:nz, :ny, :nx]
    resin = 225 + 6 * np.sin(x / 3.0) + rng.integers(-8, 9, (nz, 1, 1))     # stripes along y, steps in z
    vol = np.broadcast_to(resin, shape).astype(np.float32).copy()
    truth = np.zeros(shape, bool)
    truth[:, 12:52, 8:44] = True
    truth[4:14, 22:42, 16:36] = False                                         # a large resin pocket
    truth[16:24, 36:48, 28:40] = False                                        # a small one
    truth[10, 4, 60:62] = True                                                # a textured speck in resin
    tex = rng.uniform(60, 190, shape)
    vol[truth] = tex[truth]
    vol[16:24, 24:34, 10:22] = 150                                            # a flat pale cell
    vol[12, 40, 12:14] = 150                                                  # a flat speck in tissue
    vol[:, :10, 64:] = 0                                                      # no data
    truth[:, :10, 64:] = False
    return np.clip(np.rint(vol), 0, 255).astype(np.uint8), truth


def test_tissue_mask_is_faithful():
    vol, truth = _tissue_volume()
    mask = mt.tissue_mask(vol, rim_px=4, min_resin=1000)
    assert mask.dtype == np.uint8 and set(np.unique(mask)) <= {0, 255}
    tissue = mask > 0
    assert not tissue[vol == 0].any()                                         # no data: never tissue
    assert not tissue[5:13, 24:40, 18:34].any()                               # the hole stays open (but its
    assert not tissue[18:22, 40:44, 32:36].any()                               # edge); the small one too
    assert tissue[16:24, 24:34, 10:22].all()                                   # the pale cell is tissue
    assert not tissue[10, 4, 60:62].any()                                      # the speck in resin went
    assert tissue[12, 40, 12:14].all()                                         # the flat speck in tissue is tissue
    inner = np.zeros(vol.shape, bool)
    inner[:, 14:50, 10:42] = True
    assert tissue[inner & truth].mean() > 0.99                                 # tissue kept
    resin = ~truth & (vol > 0)
    resin[:, 10:54, 6:46] = False                                              # away from the tissue's edge
    assert not tissue[resin].any()                                             # stripes and steps: resin


def _scene(shape=(8, 12, 16), f=1, seed=1):
    rng = np.random.default_rng(seed)
    grey = rng.integers(1, 255, shape, dtype=np.uint8)
    mask = np.zeros(shape, np.uint8)
    return mt.Scene(grey, mask, f)


def _face_on(scene, z_plane, size=(16, 12), focal=1e5):
    """A camera looking along +z at the plane z = z_plane, the level's voxels 1:1 on the frame (far
    away: the planes behind it are the same size on screen)."""
    nz, ny, nx = scene.shape
    f = scene.f
    return movie.camera((f * nx / 2, f * ny / 2, z_plane), 0.0, 0.0, focal * f, 1), size, focal


def test_opaque_block_face_on_shows_the_first_slice():
    scene = _scene()
    cam, size, focal = _face_on(scene, 0.5)
    frame = mt.render_volume(scene, cam, size, focal, resin=1.0)
    np.testing.assert_allclose(frame, scene.grey[0], atol=1e-3)               # grey values unchanged
    scene.mask[:] = 255                                                        # all tissue: the same
    np.testing.assert_allclose(mt.render_volume(scene, cam, size, focal, resin=0.0), scene.grey[0], atol=1e-3)


def test_background_turns_transparent_and_no_data_is_clear():
    scene = _scene()
    scene.mask[3] = 255                                                       # only slice 3 is tissue
    scene.grey[0, :4, :4] = 0                                                  # no data in a corner of slice 0
    cam, size, focal = _face_on(scene, 0.5)
    clear = mt.render_volume(scene, cam, size, focal, resin=0.0)
    np.testing.assert_allclose(clear, scene.grey[3], atol=0.25)               # only the tissue, as it is
    block = mt.render_volume(scene, cam, size, focal, resin=1.0)
    np.testing.assert_allclose(block[4:, 4:], scene.grey[0][4:, 4:], atol=0.25)
    np.testing.assert_allclose(block[:4, :4], scene.grey[1][:4, :4], atol=0.25)   # behind no data: slice 1
    half = mt.render_volume(scene, cam, size, focal, resin=0.5)
    front = 0.5 * scene.grey[0, 6, 8] + 0.25 * scene.grey[1, 6, 8] + 0.125 * (scene.grey[2, 6, 8] + scene.grey[3, 6, 8])
    assert half[6, 8] == pytest.approx(front, abs=0.05)                        # three resin voxels, then tissue


def test_cut_plane_removes_the_volume_in_front():
    scene = _scene(f=2)
    cam, size, focal = _face_on(scene, 2 * 3 + 1.0)                           # slice 3's plane (s0 z 7)
    frame = mt.render_volume(scene, cam, size, focal, resin=1.0, cut=6.0)      # cut at slice 3's front face
    np.testing.assert_allclose(frame, scene.grey[3], atol=0.25)                # slices 0-2 gone
    part = mt.render_volume(scene, cam, size, focal, resin=1.0, cut=5.0)       # half of slice 2 left
    np.testing.assert_allclose(part, 0.5 * scene.grey[2] + 0.5 * scene.grey[3], atol=0.25)


def test_no_jump_where_the_slice_stack_switches_axis():
    scene = _scene(shape=(16, 16, 16), seed=2)
    scene.mask[4:12, 4:12, 4:12] = 255
    nz, ny, nx = scene.shape
    size, focal = (48, 48), 60.0
    frames = []
    for yaw in (44.9, 45.1):                                                   # z stack, then x stack
        cam = movie.camera((nx / 2, ny / 2, nz / 2), math.radians(yaw), math.radians(10), 60.0, 1)
        frames.append(mt.render_volume(scene, cam, size, focal, resin=0.0))
    assert np.abs(frames[0] - frames[1]).mean() < 2.0
    assert frames[0].max() > 0


def test_display_crop():
    """A crop box in s0 voxels: the voxels reaching into what is cut off go, at any level; a fly-through
    frame keeps only the columns past x0."""
    grey = np.full((4, 6, 10), 100, np.uint8)
    np.testing.assert_array_equal(mt.crop_volume(grey, {"x0": 6}, 2)[:, :, :3], 0)          # 6 s0 = 3 voxels
    assert (mt.crop_volume(grey, {"x0": 6}, 2)[:, :, 3:] == 100).all()
    np.testing.assert_array_equal(mt.crop_volume(grey, {"x0": 5}, 2)[:, :, :3], 0)          # part of voxel 2
    assert (mt.crop_volume(grey, {"y1": 4}, 2)[:, 2:] == 0).all() and mt.crop_volume(grey, {}, 2) is grey
    keep = mt.crop_in_frame({"x0": 10.0}, (8, 4), 2.0, 12.0, 0.0)               # columns at x 5, 7, ... 19
    np.testing.assert_array_equal(keep[0], [0, 0, 0, 1, 1, 1, 1, 1])


def test_partly_imaged_edge_voxels_are_left_out():
    """A coarse voxel on the imaged area's edge is darkened by the zeros averaged into it: left out (no
    data, so transparent), the voxels inside kept as they are; the volume's own faces are no edge."""
    grey = np.full((6, 8, 10), 150, np.uint8)
    grey[:, :, 7:] = 0                                                        # not imaged
    grey[:, :, 6] = 60                                                        # partly imaged
    grey[3, 4, 2] = 0                                                         # black inside: no edge
    part = mt.imaged_part(grey)
    assert not part[:, :, 6:].any() and (part[:, :, :6] == grey[:, :, :6]).all()
    grey[3, 4, 2] = 150
    assert (mt.imaged_part(grey, 0) == grey).all()
    scene = mt.Scene(part, np.full(grey.shape, 255, np.uint8), 1)              # all "tissue"
    cam, size, focal = _face_on(scene, 0.5, size=(10, 8))
    frame = mt.render_volume(scene, cam, size, focal, resin=0.0)
    np.testing.assert_allclose(frame[:, :6], 150, atol=0.25)
    assert np.abs(frame[:, 6:]).max() < 0.25                                   # not drawn black: not drawn


def _cfg_dict(tmp_path, **m):
    return {"name": "vol", "raw_dir": str(tmp_path), "output_dir": str(tmp_path / "out"),
            "tissue_movie": {**SHORT, **m},
            "movie": {"size": SIZE, "fps": 10, "slices_per_s": 2, "hold_start_s": 1.0, "zoom_s": 1.0,
                      "hold_end_s": 0.5, "target": [40, 32, 16], "view_rotation_deg": 0, "cache_gb": 0.1}}


def _write(tmp_path, **m):
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(_cfg_dict(tmp_path, **m)))
    return path


def _cfg(tmp_path, **m):
    return load_config(_write(tmp_path, **m), step_defaults=deep_merge(movie.DEFAULTS, mt.DEFAULTS))


def test_timeline_orbit_and_fade(tmp_path):
    m = _cfg(tmp_path)["tissue_movie"]
    scene = _scene(shape=(24, 64, 72))
    plan = mt.timeline(m, scene)
    assert len(plan) == 25 and (plan["part"] == "orbit").all()               # 0.5 block, 0.5 turn, 1 fade, 0.5
    yaw = plan["yaw"].to_numpy()
    assert yaw[0] == pytest.approx(-60) and yaw[10] == pytest.approx(0, abs=0.5)    # the turn ends at 1 s
    assert yaw[-1] == pytest.approx(10, abs=2) and (np.diff(yaw) > -1e-9).all()     # then drifts on 10 degrees
    step = np.diff(yaw)
    assert step[2] < step[7] and step[12] < step[7]                           # speeds up and slows down
    assert (plan["resin"].iloc[:11] == 1).all() and (plan["resin"].iloc[20:] == 0).all()
    assert (np.diff(plan["resin"]) <= 1e-12).all()                            # only ever clears
    # the block fills `fill` of the frame, centred (the push-in brings it closer later)
    focal = SIZE[0] / 2 / math.tan(math.radians(m["fov_deg"]) / 2)
    look, dist = mt.fit(scene.box(), math.radians(-60), math.radians(22), -1, SIZE, focal, 0.84)
    p = movie.project(scene.box(), movie.camera(look, math.radians(-60), math.radians(22), dist, -1), SIZE, focal)
    assert max(np.ptp(p[:, 0]) / SIZE[0], np.ptp(p[:, 1]) / SIZE[1]) == pytest.approx(0.84, abs=1e-3)
    assert (p.min(0) + p.max(0)) / 2 == pytest.approx([32, 24], abs=0.05)
    row = plan.iloc[0]
    assert (row.lx, row.ly, row.lz, row.dist) == pytest.approx((*look, dist), rel=0.05)   # (smoothed in time)
    with pytest.raises(ValueError, match="fade_out_s"):
        mt.timeline({**m, "fade_out_s": 1.0}, scene)
    with pytest.raises(ValueError, match="0 or 180"):
        mt.timeline({**m, "view_rotation_deg": 90}, scene)


def _pyramid(root, vol, num_scales=3):
    """vol as an OME-Zarr pyramid (each level the 2x mean of the previous one)."""
    data = [vol]
    shapes = omezarr.create(root, vol.shape, (8.0, 8.0, 8.0), num_scales=num_scales, chunk=(8, 32, 32),
                            shard=(8, 64, 64))
    for s, sh in enumerate(shapes):
        if s:
            prev = data[-1].astype(np.float32)
            prev = np.pad(prev, [(0, 2 * n - p) for n, p in zip(sh, prev.shape)], mode="edge")
            data.append(prev.reshape(sh[0], 2, sh[1], 2, sh[2], 2).mean(axis=(1, 3, 5)).round().astype(np.uint8))
        omezarr.open_scale(root, s)[...].write(data[s]).result()


def screen_scale(plan, focal):
    """Per row: s0 voxels per frame px on the cut plane at the frame centre (the fly-through's zoom; for
    the 3D rows, the camera's distance to the cut plane along its view / focal)."""
    out = []
    for r in plan.itertuples():
        if r.part == "fly":
            out.append(r.zoom)
            continue
        pos, forward = movie.camera((r.lx, r.ly, r.lz), math.radians(r.yaw), math.radians(r.tilt), r.dist, -1)[:2]
        cut = r.cut if np.isfinite(r.cut) else r.lz
        out.append((cut - pos[2]) / forward[2] / focal)
    return np.array(out)


def test_fly_through_ending(tmp_path):
    """The cut sweeps from z = 0 into the fly-through's scroll as the camera comes face-on to it, with no
    stop; the zoom-in starts zoom_overlap_s before (the approach blending into its moving camera), and
    from then to the zoom's end the view only ever closes in. Then the fly-through, as pipeline.movie's
    (no hold at the start), tissue only, the fine slice fading in on the cut."""
    vol, truth = _tissue_volume(shape=(48, 64, 96))
    vol[:, 12:52, 8:44] = np.where(truth[:, 12:52, 8:44], vol[:, 12:52, 8:44], 225)    # (the corner's pocket)
    _pyramid(tmp_path / "out" / "vol.ome.zarr", vol)
    for yaw, order in (([-60, 0], "reverse"), ([-120, 0], "imaging")):           # (videos 4 and 4b)
        cfg = _cfg(tmp_path, fly_through=True, level=1, trim_edge_px=0, mask={"rim_px": 4, "min_resin": 1000},
                   zoom_overlap_s=0.3, approach_s=1.0, yaw_deg=yaw, build_order=order)
        mt._WORKER.clear()
        scene, plan = mt.plan_of(cfg)
        full = mt.fly_plan(cfg, (48, 64, 96))
        fly = full[full["t"] >= 0.3 - 1e-9]                                    # (the first 0.3 s in the approach)
        parts = plan["part"].tolist()
        n_vol = parts.index("fly")
        assert parts[n_vol:] == ["fly"] * len(fly)
        np.testing.assert_allclose(plan[plan["part"] == "fly"][["zoom", "cx", "cy", "z"]].to_numpy(),
                                   fly[["zoom", "cx", "cy", "z"]].to_numpy())
        last, first = plan.iloc[n_vol - 1], plan.iloc[n_vol]
        assert last.cut < fly["z"].iloc[0] + 0.5 and last.cut > plan["cut"].iloc[n_vol - 2]
        assert (last.yaw, last.tilt) == pytest.approx((0, 0), abs=3)            # (nearly face-on)
        assert plan["resin"].iloc[n_vol - 1] == 0 and (np.diff(plan["resin"].iloc[:n_vol]) <= 1e-12).all()   # clear
        cuts = plan["cut"].iloc[:n_vol].dropna()
        assert cuts.iloc[0] == pytest.approx(0, abs=0.1) and (np.diff(cuts) > 0).all()   # from the first slice on
        # the view's scale on the cut: from the zoom-in's start (0.3 s before the hand-over) to its end, it
        # only ever closes in, and keeps moving through the hand-over
        focal = SIZE[0] / 2 / math.tan(math.radians(40) / 2)
        t_join = first.t
        span = (plan["t"] >= t_join - 0.3 - 1e-9) & (plan["t"] <= t_join + 1.0 - 0.3 + 1e-9)
        scale = screen_scale(plan[span], focal)
        assert (np.diff(scale) <= 1e-6 * scale[:-1]).all()
        step = np.diff(np.log(scale))
        assert step[2] < 0 and (step[1:-1] < 0).all()                            # no still moment
        assert first.blend == 0 and plan["blend"].iloc[-1] == 1                # the fine slice fades in
    t = np.array([0.0, 1e-4, 1 - 1e-4, 1.0])                                   # the sweep: from rest at 0 to the
    z = mt.sweep_path(t, 0.0, 1.0, 20.5, 2.0)                                  # slice, at the scroll's speed
    assert z[0] == 0 and z[-1] == pytest.approx(20.5)
    assert (z[1] - z[0]) / 1e-4 == pytest.approx(0, abs=0.01) and (z[3] - z[2]) / 1e-4 == pytest.approx(2, rel=1e-3)
    plain = movie.timeline({**mt.fly_settings(cfg), "hold_start_s": 0.0}, (48, 64, 96))
    np.testing.assert_allclose(full[["zoom", "cx", "cy", "z"]].to_numpy(), plain[["zoom", "cx", "cy", "z"]].to_numpy())
    # on the cut, tissue only: black where the mask has no tissue
    pic = mt.fly_picture(cfg, mt._WORKER, fly["zoom"].iloc[-1], fly["cx"].iloc[-1], fly["cy"].iloc[-1],
                         fly["z"].iloc[-1])
    assert pic.shape == (48, 64) and pic.max() > 0


def test_stack_grows_towards_the_camera(tmp_path):
    """The build: the volume behind a plane sweeping from the last slice to the first, the first slice
    appearing over first_slice_s, then the whole block (no cut) as the orbit goes on."""
    m = _cfg(tmp_path, build_s=1.0, first_slice_s=0.3)["tissue_movie"]
    scene = _scene(shape=(24, 64, 72))
    plan = mt.timeline(m, scene)
    build = plan[plan["part"] == "build"]
    assert len(build) == 10 and build["cut"].iloc[0] == 24 and (np.diff(build["cut"]) < 0).all()
    assert 24 - build["cut"].iloc[1] == pytest.approx(1 / 3)                  # a third of the first slice
    assert np.isnan(plan["cut"].iloc[10:]).all()
    imaging = mt.timeline({**m, "build_order": "imaging"}, scene)                # as imaged: z = 0 first
    assert np.isnan(imaging["cut"].iloc[:10]).all() and imaging["back"].iloc[1] == pytest.approx(1 / 3)
    small = _scene()
    cam, size, focal = _face_on(small, 0.5)
    np.testing.assert_allclose(mt.render_volume(small, cam, size, focal, 1.0, back=0.5), 0.5 * small.grey[0],
                               atol=0.25)                                      # half of the first slice
    np.testing.assert_allclose(mt.render_volume(small, cam, size, focal, 1.0, cut=7.5), 0.5 * small.grey[7],
                               atol=0.25)                                      # half of the last slice


def _cfg_dict(tmp_path, **m):
    return {"name": "vol", "raw_dir": str(tmp_path), "output_dir": str(tmp_path / "out"),
            "tissue_movie": {**SHORT, **m},
            "movie": {"size": SIZE, "fps": 10, "slices_per_s": 2, "hold_start_s": 1.0, "zoom_s": 1.0,
                      "hold_end_s": 0.5, "target": [40, 32, 16], "view_rotation_deg": 0, "cache_gb": 0.1}}


def _write(tmp_path, **m):
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(_cfg_dict(tmp_path, **m)))
    return path


def _cfg(tmp_path, **m):
    return load_config(_write(tmp_path, **m), step_defaults=deep_merge(movie.DEFAULTS, mt.DEFAULTS))


def test_frames_and_encode(tmp_path):
    vol, _ = _tissue_volume(shape=(32, 64, 72))
    _pyramid(tmp_path / "out" / "vol.ome.zarr", vol)
    path = _write(tmp_path)
    assert mt.main(["--config", str(path), "mask"]) == 0
    assert (tmp_path / "out" / "qc" / "vol_tissue_movie_mask.png").exists()
    out = tmp_path / "out" / "work" / "movie" / "vol_tissue_movie"
    assert mt.main(["--config", str(path), "frames", "--frames", "0,12"]) == 0
    assert sorted(p.name for p in out.glob("f*.png")) == ["f00000.png", "f00012.png"]
    first = cv2.imread(str(out / "f00000.png"), cv2.IMREAD_UNCHANGED)
    assert first.shape == (48, 64) and first.dtype == np.uint8 and first.max() > 0
    cfg = load_config(path, step_defaults=deep_merge(movie.DEFAULTS, mt.DEFAULTS))
    with pytest.raises(RuntimeError, match="23 of 25 frames missing"):
        mt.encode(cfg)
    assert mt.main(["--config", str(path), "frames", "--workers", "2"]) == 0
    assert len(list(out.glob("f*.png"))) == 25
    stamp = (out / "f00000.png").stat().st_mtime_ns
    assert mt.main(["--config", str(path), "frames"]) == 0                    # resumes: nothing redone
    assert (out / "f00000.png").stat().st_mtime_ns == stamp
    assert mt.main(["--config", str(_write(tmp_path, yaw_deg=[-20, 30])), "frames", "--frames", "3"]) == 0
    assert [p.name for p in out.glob("f*.png")] == ["f00003.png"]             # changed rows are redone
    # a preview: half the size, the same view, its own frames
    assert mt.main(["--config", str(path), "frames", "--preview", "2", "--frames", "5"]) == 0
    small = cv2.imread(str(out / "preview2" / "f00005.png"), cv2.IMREAD_UNCHANGED)
    assert small.shape == (24, 32) and [p.name for p in out.glob("f*.png")] == ["f00003.png"]
    if shutil.which("ffmpeg"):
        assert mt.main(["--config", str(path), "frames"]) == 0
        assert mt.main(["--config", str(path), "encode"]) == 0
        assert (tmp_path / "out" / "vol_tissue_movie.mp4").stat().st_size > 1000


def test_volume_stored_rotated_180_degrees(tmp_path):
    """A volume stored rotated 180 degrees in x-y (V' = V[:, ::-1, ::-1], as render.rotate_deg 180 writes
    it) filmed with view_rotation_deg 0, the yaws, drifts and spin negated and the x-y settings mapped
    (X -> nx - X: the target, the crop), gives the movie filmed from V with view_rotation_deg 180:
    the build, the tissue alone, the cut and the fly-through (the frames as shown)."""
    vol, truth = _tissue_volume(shape=(48, 64, 96))
    vol[:, 12:52, 8:44] = np.where(truth[:, 12:52, 8:44], vol[:, 12:52, 8:44], 225)
    nz, ny, nx = vol.shape
    common = {**SHORT, "fly_through": True, "level": 1, "trim_edge_px": 0, "build_s": 1.0, "turn_s": 1.0,
              "mask": {"rim_px": 4, "min_resin": 1000}, "zoom_overlap_s": 0.3, "approach_s": 1.0,
              "zoom_out_s": 1.0, "spin_s": 1.0, "spin_in_s": 0.3, "pull_min_s": 0.5, "spin_overlap_s": 0.5}
    tx, ty, tz = 40, 30, 16
    variants = {
        "a": (vol, {"view_rotation_deg": 180, "yaw_deg": [-60, 0], "drift_deg": [3, 0], "spin_deg": 360,
                    "crop": {"x0": 8}}, [tx, ty, tz]),
        "b": (vol[:, ::-1, ::-1], {"view_rotation_deg": 0, "yaw_deg": [60, 0], "drift_deg": [-3, 0],
                                   "spin_deg": -360, "crop": {"x1": nx - 8}}, [nx - tx, ny - ty, tz]),
    }
    frames = {}
    for key, (data, m, target) in variants.items():
        root = tmp_path / key
        _pyramid(root / "out" / "vol.ome.zarr", np.ascontiguousarray(data))
        cfg_dict = {"name": "vol", "raw_dir": str(root), "output_dir": str(root / "out"),
                    "tissue_movie": {**common, **m},
                    "movie": {"size": SIZE, "fps": 10, "slices_per_s": 2, "hold_start_s": 1.0, "zoom_s": 1.0,
                              "hold_end_s": 0.5, "target": target, "view_rotation_deg": m["view_rotation_deg"],
                              "cache_gb": 0.1}}
        (root / "cfg.yaml").write_text(yaml.safe_dump(cfg_dict))
        cfg = load_config(root / "cfg.yaml", step_defaults=deep_merge(movie.DEFAULTS, mt.DEFAULTS))
        mt._WORKER.clear()
        scene, plan = mt.plan_of(cfg)
        parts = plan["part"].to_numpy()
        picks = {"build": int(np.argmax(parts == "build")) + 5,
                 "fading": int(np.argmax(plan["resin"].to_numpy() < 0.5)),
                 "cut": int(np.flatnonzero(np.isfinite(plan["cut"].to_numpy()) & (parts == "orbit"))[3]),
                 "fly": int(np.argmax(parts == "fly")) + 8,
                 "spin": len(plan) - 6}
        frames[key] = {name: mt.render_frame(cfg, SimpleNamespace(**plan.iloc[k].to_dict())).astype(float)
                       for name, k in picks.items()}
        frames[key]["n"] = len(plan)
    assert frames["a"]["n"] == frames["b"]["n"]
    for name in ("build", "fading", "cut", "fly", "spin"):
        a, b = frames["a"][name], frames["b"][name]
        assert a.max() > 0, name
        assert np.abs(a - b).mean() < 1.0, name                                # up to resampling
        assert np.corrcoef(a.ravel(), b.ravel())[0, 1] > 0.99, name


def test_twice_the_frame_size_is_the_same_movie(tmp_path):
    """At twice the frame size with zoom_end halved (e.g. 4K from 1080p), the timeline is the same: the
    same camera, cut and scroll (its speeds per zoom on a ref_width frame), zooms halved."""
    vol, _ = _tissue_volume(shape=(48, 64, 96))
    _pyramid(tmp_path / "out" / "vol.ome.zarr", vol)
    plans = []
    for k in (1, 2):
        cfg = _cfg(tmp_path, fly_through=True, level=1, trim_edge_px=0, mask={"rim_px": 4, "min_resin": 1000},
                   zoom_overlap_s=0.3, approach_s=1.0, zoom_out_s=1.0, spin_s=1.0, spin_in_s=0.3, pull_min_s=0.5,
                   spin_overlap_s=0.5, size=[64 * k, 48 * k])
        cfg["movie"] = {**cfg["movie"], "size": [64 * k, 48 * k], "zoom_end": 1.0 / k}
        mt._WORKER.clear()
        plans.append(mt.plan_of(cfg)[1])
    a, b = plans
    assert len(a) == len(b)
    for col in ("yaw", "tilt", "lx", "ly", "lz", "dist", "resin", "cut", "cx", "cy", "z"):
        np.testing.assert_allclose(a[col].to_numpy(float), b[col].to_numpy(float), rtol=1e-6, atol=1e-6, err_msg=col)
    np.testing.assert_allclose(b["zoom"].to_numpy(float), a["zoom"].to_numpy(float) / 2, rtol=1e-9)


def test_mask_from_another_volume(tmp_path):
    """With mask_volume, the tissue mask (and what counts as imaged) comes from that volume and the grey
    values shown from the movie's: a CLAHE-like version with texture in its resin keeps the plain
    volume's mask."""
    plain, _ = _tissue_volume(shape=(32, 64, 72))
    rng = np.random.default_rng(3)
    clahe = np.clip(plain.astype(int) + rng.integers(-25, 26, plain.shape), 1, 255).astype(np.uint8)
    clahe[plain == 0] = 0
    _pyramid(tmp_path / "out" / "plain.ome.zarr", plain)
    _pyramid(tmp_path / "out" / "clahe.ome.zarr", clahe)
    mask_settings = {"rim_px": 4, "min_resin": 1000}
    cfg = _cfg(tmp_path, volume="clahe.ome.zarr", mask_volume="plain.ome.zarr", trim_edge_px=0, mask=mask_settings)
    scene = mt.load_scene(cfg)
    np.testing.assert_array_equal(scene.mask, mt.tissue_mask(plain, **mt.DEFAULTS["tissue_movie"]["mask"] | mask_settings))
    shown = np.asarray(scene.grey)
    assert ((shown > 0) == (plain > 0)).all() and (shown[plain > 0] == np.maximum(clahe[plain > 0], 1)).all()
    own = _cfg(tmp_path, volume="clahe.ome.zarr", name="own", trim_edge_px=0, mask=mask_settings)
    assert (np.asarray(mt.load_scene(own).mask) > 0).sum() > (np.asarray(scene.mask) > 0).sum()   # resin kept
