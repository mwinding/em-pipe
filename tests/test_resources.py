"""pipeline/resources.py: Slurm resources from the dataset's size (slurm.auto), explicit settings winning."""

import pytest

import synth
from pipeline import config, resources

# The three datasets the rules were set on: sizes from their raw files, hand-set values from their configs.
SMALL = {"slices": 487, "tiles": 9, "tile_px": [8688, 9333], "tile_images": 4383, "raw_gpx": 355.4, "canvas_mpx": 693}
LONG = {"slices": 17927, "tiles": 9, "tile_px": [12000, 15000], "tile_images": 65966, "raw_gpx": 9710.0,
        "canvas_mpx": 720}
DECURTAINED = {"destreak": {"enabled": True, "reference_from": "a", "reference_to": "b"},
               "intensity": {"balance_z_distance": 1}}


def hours(t):
    return int(t.split(":")[0])


def gb(m):
    return int(m.rstrip("G"))


def test_facts_from_headers():
    shapes = {"M11_D12_tile0-0.tif": (175, 100, 120), "M11_D12_tile0-1.tif": (175, 100, 120),
              "M11_D13_tile0-0.tif": (312, 100, 120), "M11_D13_tile0-1.tif": (310, 100, 120),
              "Approach/M11_D11_tile0-0.tif": (40, 200, 200)}
    groups = {f: (11, int(f.split("_D")[1][:2]), 1) for f in shapes}
    f = resources.facts(shapes, groups)
    assert f["slices"] == 40 + 175 + 312 and f["tiles"] == 2 and f["tile_px"] == [200, 200]
    assert f["tile_images"] == 40 + 175 + 175 + 312 + 310
    assert f["raw_gpx"] == pytest.approx((40 * 40000 + 972 * 12000) / 1e9, abs=0.05)
    assert f["canvas_mpx"] == pytest.approx(0.95 * 0.04)


def test_small_volume_matches_the_hand_set_values():
    a = resources.auto({"detected": SMALL, **DECURTAINED})
    # one task per unit of work: 50-slice preview chunks, 100-slice align chunks, 64-plane render slabs
    assert (a["preview"]["array"], a["align"]["array"], a["render"]["array"], a["pyramid"]["array"]) == (10, 5, 8, 10)
    assert a["zrelations"]["array"] == 5 and a["refstats"]["array"] == 5
    # memory: at least what the runs used (render 71 GB, refstats with decurtaining 6 GB), at most the hand-set
    assert 96 <= gb(a["render"]["mem"]) <= 160 and 32 <= gb(a["refstats"]["mem"]) <= 48
    # time: the longest task of the runs fits with room (render 2.7 h, refstats 0.3 h, align 0.64 h)
    assert 5 <= hours(a["render"]["time"]) <= 24 and hours(a["refstats"]["time"]) >= 1 and hours(a["align"]["time"]) >= 2
    assert a["render"]["cpus"] == 8 and a["check"] == {"cpus": 4, "time": "02:00:00", "mem": "16G"}
    assert set(a) == set(resources.JOBS) and "array" not in a["intensity"]


def test_long_volume_gets_more_tasks_and_time():
    a = resources.auto({"detected": LONG, **DECURTAINED})
    assert a["preview"]["array"] == 40 and a["align"]["array"] == 40 and a["render"]["array"] == 60
    assert a["refstats"]["array"] == 80 and a["pyramid"]["array"] == 20
    assert 20 <= hours(a["render"]["time"]) <= 30          # the runs' longest decurtained render task: 10 h
    assert a["intensity"] == {"cpus": 2, "time": "02:00:00", "mem": "32G"}
    plain = resources.auto({"detected": LONG})
    assert plain["render"]["array"] == 30 and hours(plain["render"]["time"]) < hours(a["render"]["time"])
    assert gb(plain["refstats"]["mem"]) == 32 and gb(a["refstats"]["mem"]) >= 40   # whole 180 Mpx tiles, with EsB


def test_missing_facts_are_named():
    with pytest.raises(ValueError, match="detected: section.*missing: slices, tiles"):
        resources.auto({"detected": {"tile_px": [1, 1], "tile_images": 1, "raw_gpx": 1}})


def test_explicit_settings_win_key_by_key(tmp_path):
    path = synth.write_config(tmp_path / "c.yaml", tmp_path / "raw", tmp_path / "out", detected=SMALL,
                              slurm={"auto": True, "render": {"time": "36:00:00"}, "preview": {"array": 3}})
    cfg = config.load_config(path)
    r = config.job_options(cfg, "render")
    assert r["time"] == "36:00:00" and r["array"] == 8 and r["mem"] == resources.auto(cfg)["render"]["mem"]
    assert config.job_options(cfg, "preview")["array"] == 3
    args = config.sbatch_args(cfg, "render")
    assert "--time=36:00:00" in args and "--array=0-7" in args and "--cpus-per-task=8" in args
    # without auto nothing changes: the config's values and config.py's defaults
    path = synth.write_config(tmp_path / "d.yaml", tmp_path / "raw", tmp_path / "out", slurm={"render": {"mem": "8G"}})
    cfg = config.load_config(path)
    assert config.job_options(cfg, "render") == {"array": 20, "mem": "8G"}
    assert config.sbatch_args(cfg, "align")[1] == "--array=0-19"
