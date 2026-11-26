"""pipeline/preflight.py: the checks of a config before anything is submitted."""

from pathlib import Path

import numpy as np
import yaml

import synth
from pipeline import config, preflight

ROOT = Path(__file__).resolve().parents[1]


def raw_with(tmp_path, *names, shape=(2, 8, 8)):
    raw = tmp_path / "raw"
    for name in names:
        (raw / name).parent.mkdir(parents=True, exist_ok=True)
        synth.write_imagej(raw / name, np.zeros(shape, np.uint16), [f"s{i}" for i in range(shape[0])])
    raw.mkdir(exist_ok=True)
    return raw


def cfg(tmp_path, raw=None, **sections):
    raw = raw if raw is not None else raw_with(tmp_path, "M10_D06_tile0-0.tif")
    return synth.write_config(tmp_path / "c.yaml", raw, tmp_path / "out", **sections)


def test_example_config_has_no_unknown_keys():
    user = yaml.safe_load((ROOT / "configs" / "example.yaml").read_text())
    assert preflight.unknown_keys(user) == []


def test_plain_config_passes(tmp_path):
    res = preflight.validate(cfg(tmp_path))
    assert res.ok and not res.warnings, res.text()


def test_unknown_keys_in_every_section_with_a_suggestion(tmp_path):
    path = cfg(tmp_path, intensity={"balance_z_distnace": 1, "reference": {"medain": 150}},
               render={"clahe": {"clip_limt": 1.5}}, stich={"coarse_factor": 4}, selection={"zstart": 3},
               slurm={"rendr": {"array": 2}, "render": {"memory": "8G"}},
               stitch={"segment_modes": {12: "fixed"}}, check={"known_issues": [{"file": "a", "note": "b"}]})
    res = preflight.validate(path)
    text = "\n".join(res.errors)
    for wrong, right in (("intensity.balance_z_distnace", "balance_z_distance"), ("intensity.reference.medain", "median"),
                         ("render.clahe.clip_limt", "clip_limit"), ("'stich'", "stitch"), ("selection.zstart", "z_start"),
                         ("slurm.rendr", "render"), ("slurm.render.memory", "mem")):
        line = next(m for m in res.errors if wrong in m)
        assert f"did you mean {right!r}" in line, line
    assert "segment_modes" not in text and "known_issues" not in text   # free groups are not checked key by key
    assert len(res.errors) == 7


def test_unknown_key_in_a_base_config_is_found(tmp_path):
    base = cfg(tmp_path, preview={"chunk_slcies": 10})
    child = tmp_path / "child.yaml"
    child.write_text("base: c.yaml\nrender:\n  blend: cut\n")
    res = preflight.validate(child)
    assert not res.ok and "preview.chunk_slcies" in res.errors[0]
    assert base.exists()


def test_settings_that_need_each_other(tmp_path):
    def errors(**sections):
        return "\n".join(preflight.validate(cfg(tmp_path, **sections)).errors)

    # z-balance: overlaps on, balance from the overlaps, per_slice off
    assert "per_slice: false" in errors(intensity={"balance_z_distance": 1, "balance_source": "overlaps"},
                                        overlaps={"enabled": True})
    assert "overlaps.enabled" in errors(intensity={"balance_z_distance": 1, "balance_source": "overlaps",
                                                   "per_slice": False})
    assert "balance_source: overlaps" in errors(intensity={"balance_z_distance": 1, "per_slice": False},
                                                overlaps={"enabled": True})
    ok = dict(intensity={"balance_z_distance": 1, "balance_source": "overlaps", "per_slice": False},
              overlaps={"enabled": True})
    assert errors(**ok) == ""
    # patchgrid needs the patch bins
    assert "overlaps.patch_grid > 0" in errors(patchgrid={"enabled": True}, overlaps={"enabled": True},
                                               intensity={"balance_source": "overlaps"})
    assert errors(patchgrid={"enabled": True}, overlaps={"enabled": True, "patch_grid": 8},
                  intensity={"balance_source": "overlaps"}) == ""
    # window: reference needs the mask, the statistics and the standard's two numbers
    e = errors(intensity={"window": "reference"})
    assert "refmask.enabled" in e and "refstats.enabled" in e and "intensity.reference.median" in e
    assert errors(intensity={"window": "reference", "reference": {"median": 150, "contrast": 130}},
                  refmask={"enabled": True}, refstats={"enabled": True}) == ""
    assert "refmask.enabled" in errors(refstats={"enabled": True})
    # values
    assert "intensity.window must be one of" in errors(intensity={"window": "refrence"})
    assert "render.blend" in errors(render={"blend": "sharp"})
    assert "quoted string" in errors(slurm={"render": {"time": 14400}})
    assert "upload.destination" in errors(upload={"enabled": True}, precomputed={"enabled": True})


def test_raw_dir_messages(tmp_path):
    res = preflight.validate(cfg(tmp_path, raw=tmp_path / "nowhere"))
    assert not res.ok and "does not exist" in res.errors[0]
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "notes.txt").write_text("x")
    res = preflight.validate(cfg(tmp_path, raw=empty))
    assert "no file matching raw.file_pattern" in res.errors[0] and "notes.txt" in res.errors[0]
    assert preflight.validate(cfg(tmp_path, raw=empty), raw=False).ok   # settings only


def test_unreadable_config_and_missing_paths(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("raw_dir: [unclosed\n")
    assert "cannot be read" in preflight.validate(bad).errors[0]
    assert "not found" in preflight.validate(tmp_path / "none.yaml").errors[0]
    nopath = tmp_path / "nopath.yaml"
    nopath.write_text("name: x\n")
    assert "'raw_dir' is not set" in preflight.validate(nopath).errors[0]


def test_command_line_prints_messages_without_traceback(tmp_path, capsys):
    path = cfg(tmp_path, intensity={"windw": "median"})
    assert preflight.main([str(path)]) == 1
    err = capsys.readouterr().err
    assert "ERROR: unknown key 'intensity.windw' (did you mean 'window'?)" in err and "Traceback" not in err
    assert "nothing was submitted" in err
    assert config._main(["--config", str(path), "--validate"]) == 1
    assert preflight.main([str(cfg(tmp_path))]) == 0
    assert capsys.readouterr().out.strip().endswith("OK")


REF = {"enabled": True, "reference_from": r"^(.*/)?(M\d\d_D\d\d)(.*)$", "reference_to": r"\1Backscatter/\2_EsB\3"}


def test_esb_files_are_checked_when_decurtaining_with_a_reference(tmp_path):
    raw = raw_with(tmp_path, "M10_D06_tile0-0.tif", "M10_D07_tile0-0.tif", "Backscatter/M10_D06_EsB_tile0-0.tif")
    res = preflight.validate(cfg(tmp_path, raw=raw, destreak=REF))
    assert not res.ok
    assert "1 of 2 EsB files present; 1 missing: M10_D07 (1 file)" in res.errors[0]
    assert "destreak.enabled: false" in res.errors[0]
    # complete: passes, and says so
    raw_with(tmp_path, "Backscatter/M10_D07_EsB_tile0-0.tif")
    res = preflight.validate(cfg(tmp_path, raw=raw, destreak=REF))
    assert res.ok and "2 of 2 EsB files present" in res.notes[0]
    # stated: without decurtaining nothing is checked
    raw_with(tmp_path, "M10_D08_tile0-0.tif")
    assert preflight.validate(cfg(tmp_path, raw=raw, destreak={**REF, "enabled": False})).ok
    assert not preflight.validate(cfg(tmp_path, raw=raw, destreak=REF)).ok


def test_filter_only_destreak_is_a_stated_choice(tmp_path):
    raw = raw_with(tmp_path, "M10_D06_tile0-0.tif", "Backscatter/M10_D06_EsB_tile0-0.tif")
    res = preflight.validate(cfg(tmp_path, raw=raw, destreak={"enabled": True}))
    assert res.ok and len(res.warnings) == 2   # filter-only without saying so; and EsB files are there, unused
    assert "filter-only" in res.warnings[0] and "holds 1 EsB files" in res.warnings[1]
    res = preflight.validate(cfg(tmp_path, raw=raw, destreak={"enabled": True, "mode": "filter"}))
    assert res.ok and not res.warnings
    res = preflight.validate(cfg(tmp_path, raw=raw, destreak={"enabled": True, "mode": "reference"}))
    assert not res.ok and "destreak.mode: reference needs" in res.errors[0]
    res = preflight.validate(cfg(tmp_path, raw=raw, destreak={**REF, "mode": "filter"}))
    assert not res.ok and "mode: filter, but" in res.errors[0]
