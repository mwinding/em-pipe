"""movie: fly-through frames from an OME-Zarr pyramid, and the MP4."""

import shutil

import cv2
import numpy as np
import pytest

from pipeline import movie, omezarr
from pipeline.config import load_config

SIZE = [64, 48]
MOVIE = {"size": SIZE, "fps": 10, "slices_per_s": 2, "duration_s": 4, "hold_start_s": 1.0, "hold_end_s": 1.0,
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
