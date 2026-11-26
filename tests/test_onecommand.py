"""The one-command run (em-pipe RAW_DIR OUTPUT_DIR): the config it writes, what stops it, the chains it submits."""

import os
import re
import shutil
import sys
from pathlib import Path

import pytest
import yaml

import synth
from pipeline import config, curtains, onecommand, preflight
from test_run_pipeline import FAKE_SBATCH, calls

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def raw(tmp_path):
    """A 2 x 2 dataset over two days with the EsB file of every raw file in Backscatter/."""
    t = synth.make_dataset(tmp_path / "raw", n_slices=16, tile_shape=(64, 80), overlap=(8, 10))
    (tmp_path / "raw" / "Backscatter").mkdir()
    for f in sorted((tmp_path / "raw").glob("M*.tif")):
        shutil.copy(f, f.parent / "Backscatter" / f.name.replace("_tile", "_EsB_tile"))
    return t.raw_dir


@pytest.fixture
def sbatch(tmp_path, monkeypatch):
    """A fake sbatch on PATH (as test_run_pipeline's); returns its log file."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "sbatch").write_text(FAKE_SBATCH)
    (bin_dir / "sbatch").chmod(0o755)
    log = tmp_path / "sbatch.log"
    monkeypatch.setenv("PATH", os.pathsep.join([str(bin_dir), str(Path(sys.executable).parent), os.environ["PATH"]]))
    monkeypatch.setenv("EM_PIPE_SKIP_ENV", "1")
    monkeypatch.setenv("SBATCH_LOG", str(log))
    return log


def run(*args):
    return onecommand.main([str(a) for a in args])


def test_dry_run_shows_everything_and_changes_nothing(tmp_path, raw, sbatch, capsys):
    out = tmp_path / "out"
    assert run(raw, out, "--name", "vol", "--dry-run") == 0
    text = capsys.readouterr().out
    assert not out.exists() and not sbatch.exists()
    assert "8 of 8 EsB files present" in text and "curtain direction not measured" in text   # tiles under 256 px
    assert f"--- {out / 'config.yaml'} (not written: --dry-run) ---" in text
    assert "reference_to: '\\1Backscatter/\\2_EsB\\3'" in text and "direction: vertical" in text
    assert "Slurm resources per job" in text and re.search(r"render +array 1 +cpus 8 +mem 64G +time 01:00:00", text)
    assert "Decisions that stay with a person" in text
    lines = [line for line in text.splitlines() if line.startswith("sbatch ")]
    names = [c["job-name"] for c in calls(lines)]
    assert names[0] == "em-check" and "em-refstats-run" in names and "em-zrelations-run" in names
    plain = names[:names.index("em-qc-summary") + 1]
    assert plain[-3:] == ["em-quality-run", "em-quality-merge", "em-qc-summary"]
    assert plain[plain.index("em-intensity") + 1] == "em-render-init"   # no step between intensity and render
    clahe = calls(lines)[len(plain):]
    assert [c["job-name"] for c in clahe][:2] == ["em-render-init", "em-render-run"]
    assert clahe[0]["dependency"] == "afterok:<em-intensity>" and clahe[0]["args"][0] == str(out / "config_clahe.yaml")
    assert clahe[-1]["job-name"] == "em-qc-summary" and clahe[-1]["dependency"].endswith(":<em-qc-summary>")
    assert str(tmp_path / "raw") in text and "/tmp" not in text.replace(str(tmp_path), "")   # no temporary paths shown


def test_run_writes_the_config_and_submits_both_chains(tmp_path, raw, sbatch, capsys):
    out = tmp_path / "out"
    assert run(raw, out, "--name", "vol") == 0
    text = capsys.readouterr().out
    user = yaml.safe_load((out / "config.yaml").read_text())
    assert user["base"] == str(ROOT / "configs" / "presets" / "standard.yaml")
    assert user["name"] == "vol" and user["raw_dir"] == str(raw) and user["output_dir"] == str(out)
    assert user["destreak"] == {"reference_from": r"^(.*/)?(M\d\d_D\d\d)(.*)$", "reference_to": r"\1Backscatter/\2_EsB\3",
                                "direction": "vertical"}
    assert user["detected"]["slices"] == 16 and user["detected"]["tiles"] == 4 and user["detected"]["tile_px"] == [64, 80]
    assert user["refstats"] == {"every": 32} and user["render"] == {"name": "vol.ome.zarr"}
    # the written configs pass the checks every run starts with, and resolve to the preset's settings
    assert preflight.validate(out / "config.yaml").ok and preflight.validate(out / "config_clahe.yaml").ok
    cfg = config.load_config(out / "config.yaml", step_defaults=config.step_defaults())
    assert cfg["intensity"]["window"] == "reference" and cfg["intensity"]["balance_z_distance"] == 1
    assert cfg["intensity"]["reference"]["median"] == 150 and cfg["intensity"]["reference"]["contrast"] == 130
    assert cfg["destreak"]["enabled"] and cfg["destreak"]["mode"] == "reference" and cfg["destreak"]["levels"] == 8
    assert "zcorrect" not in cfg and not cfg["render"]["clahe"]["enabled"] and cfg["render"]["blend"] == "cut"
    clahe = config.load_config(out / "config_clahe.yaml", step_defaults=config.step_defaults())
    assert clahe["render"]["clahe"] == {"enabled": True, "clip_limit": 1.5, "tile_px": 256}
    assert clahe["render"]["name"] == "vol_clahe.ome.zarr"
    jobs = calls(sbatch.read_text().splitlines())
    names = [j["job-name"] for j in jobs]
    k = names.index("em-qc-summary")
    intensity_id = 1000 + names.index("em-intensity") + 1
    assert names[k + 1] == "em-render-init" and jobs[k + 1]["dependency"] == f"afterok:{intensity_id}"
    assert jobs[-1]["job-name"] == "em-qc-summary" and jobs[-1]["dependency"].endswith(f":{1000 + k + 1}")
    assert "summary_vol.html" in text and (out / "qc" / "em_pipe_detected.json").exists()
    # again: the same config is kept and the chain can be submitted again; another name is refused
    assert run(raw, out, "--name", "vol", "--to", "check") == 0
    assert "config.yaml exists and says the same: kept" in capsys.readouterr().out
    assert run(raw, out, "--name", "other", "--dry-run") == 1
    assert "exists and says something else: nothing was changed" in capsys.readouterr().out


def test_missing_esb_stops_before_anything_is_written(tmp_path, raw, sbatch, capsys):
    out = tmp_path / "out"
    for f in (raw / "Backscatter").glob("M09_D25*"):
        f.unlink()
    assert run(raw, out) == onecommand.STOP
    text = capsys.readouterr().out
    assert "4 of 8 EsB files present; 4 missing: M09_D25 (4 files)" in text and "--no-esb" in text
    assert "STOPPED (EsB files missing or not matching): nothing was written or submitted" in text
    assert not out.exists() and not sbatch.exists()
    # stated: without decurtaining
    assert run(raw, out, "--no-esb", "--no-clahe") == 0
    user = yaml.safe_load((out / "config.yaml").read_text())
    assert user["destreak"] == {"enabled": False} and not (out / "config_clahe.yaml").exists()
    assert "decurtaining: off (--no-esb)" in capsys.readouterr().out
    names = [j["job-name"] for j in calls(sbatch.read_text().splitlines())]
    assert names[-1] == "em-qc-summary" and names.count("em-render-run") == 1


def test_conflicting_curtain_direction_stops_until_a_person_states_it(tmp_path, raw, sbatch, capsys, monkeypatch):
    out = tmp_path / "out"
    verdict = curtains.decide([{"vote": "vertical", "angle": 0.0, "amplitude": 0.01}] * 6
                              + [{"vote": "horizontal", "angle": 90.0, "amplitude": 0.01}] * 6)
    monkeypatch.setattr(curtains, "detect", lambda *a, **k: verdict)
    assert run(raw, out) == onecommand.STOP
    text = capsys.readouterr().out
    assert "curtain direction conflicting" in text and "--direction" in text and not out.exists() and not sbatch.exists()
    assert run(raw, out, "--direction", "horizontal", "--to", "intensity") == 0
    assert yaml.safe_load((out / "config.yaml").read_text())["destreak"]["direction"] == "horizontal"
    text = capsys.readouterr().out
    assert "stated with --direction, not measured" in text and "the CLAHE volume was not submitted (--to intensity)" in text
    names = [j["job-name"] for j in calls(sbatch.read_text().splitlines())]
    assert names[-1] == "em-intensity" and "em-render-init" not in names
    monkeypatch.setattr(curtains, "detect", lambda *a, **k: {**verdict, "direction": "horizontal", "use": "horizontal",
                                                             "message": "curtains are horizontal: 12 of 12 windows"})
    assert run(raw, tmp_path / "out2", "--dry-run") == 0
    assert "direction: horizontal" in capsys.readouterr().out


def test_raw_folder_problems_are_messages(tmp_path, sbatch, capsys):
    assert run(tmp_path / "nowhere", tmp_path / "out") == 1
    assert "does not exist" in capsys.readouterr().out
    (tmp_path / "empty").mkdir()
    (tmp_path / "empty" / "image_001.tif").touch()
    assert run(tmp_path / "empty", tmp_path / "out") == 1
    text = capsys.readouterr().out
    assert "no file in" in text and "image_001.tif" in text and not (tmp_path / "out").exists()


def test_the_preset_is_valid_and_lab_neutral():
    preset = ROOT / "configs" / "presets" / "standard.yaml"
    user = yaml.safe_load(preset.read_text())
    assert preflight.unknown_keys(user) == []
    assert not {"raw_dir", "output_dir", "name"} & set(user)          # settings only: no paths, no names
    assert "zcorrect" not in user and user["slurm"] == {"auto": True}   # the removed step is not named any more
    assert (ROOT / "em-pipe").stat().st_mode & 0o111                    # the entry script is executable


def test_image_files_that_would_be_left_out_stop_the_run(tmp_path, raw, sbatch, capsys):
    """A stack whose name the file pattern does not match is data left out: stated, never silent."""
    import numpy as np
    synth.write_imagej(raw / "M09_D24D25_tile0-0.tif", np.zeros((2, 8, 8), np.uint16), ["a", "b"])
    out = tmp_path / "out"
    assert run(raw, out, "--dry-run") == onecommand.STOP
    text = capsys.readouterr().out
    assert "1 image files" in text and "M09_D24D25_tile0-0.tif" in text and "--skip-unmatched" in text
    assert run(raw, out, "--dry-run", "--skip-unmatched") == 0
    assert "left out, as stated with --skip-unmatched" in capsys.readouterr().out
    # both reasons to stop are reported together: the EsB report is not hidden behind the first one
    for f in (raw / "Backscatter").glob("M09_D25*"):
        f.unlink()
    assert run(raw, out, "--dry-run") == onecommand.STOP
    text = capsys.readouterr().out
    assert "4 missing: M09_D25 (4 files)" in text
    assert "STOPPED (image files that would be left out; EsB files missing or not matching)" in text


def test_file_pattern_for_other_names(tmp_path, sbatch, capsys):
    """Raw files without a tile in their name (a single-tile phase) are part of the volume with the pattern given."""
    import numpy as np
    raw = tmp_path / "raw"
    for name in ("M10_D16_part1.tif", "Backscatter/M10_D16_EsB_part1.tif"):
        (raw / name).parent.mkdir(parents=True, exist_ok=True)
        synth.write_imagej(raw / name, np.zeros((3, 8, 8), np.uint16), ["a", "b", "c"])
    out = tmp_path / "out"
    assert run(raw, out, "--dry-run") == 1 and "no file in" in capsys.readouterr().out
    pattern = r"^M(?P<month>\d{2})_D(?P<day>\d{2})(?:_tile(?P<row>\d+)-(?P<col>\d+))?(?:_part(?P<part>\d+))?\.tif$"
    assert run(raw, out, "--file-pattern", pattern, "--to", "check") == 0
    user = yaml.safe_load((out / "config.yaml").read_text())
    assert user["raw"] == {"file_pattern": pattern} and user["detected"]["slices"] == 3
    assert "1 of 1 EsB files present" in capsys.readouterr().out
    assert run(raw, out, "--file-pattern", "M(", "--dry-run") == 1
