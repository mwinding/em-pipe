"""run_pipeline.sh and `python -m pipeline.config`: the sbatch chain, with a fake sbatch on PATH."""

import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

import synth
from pipeline import config

ROOT = Path(__file__).resolve().parents[1]
CHAIN = ["em-check", "em-preview-run", "em-preview-merge", "em-stitch-run", "em-stitch-merge", "em-align-run",
         "em-align-solve", "em-intensity", "em-render-init", "em-render-run", "em-pyramid-s1", "em-pyramid-s2", "em-export",
         "em-quality-run", "em-quality-merge", "em-qc-summary"]
FAKE_SBATCH = """#!/bin/bash
# Records its arguments (one call per line) and prints a job id like sbatch --parsable.
echo "$*" >> "$SBATCH_LOG"
n=0; while read -r _; do n=$((n + 1)); done < "$SBATCH_LOG"
echo "$((1000 + n));cluster"
"""


@pytest.fixture
def env(tmp_path):
    """Environment with a fake sbatch and this python on PATH and env.sh skipped; returns (env, sbatch log)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "sbatch").write_text(FAKE_SBATCH)
    (bin_dir / "sbatch").chmod(0o755)
    log = tmp_path / "sbatch.log"
    path = os.pathsep.join([str(bin_dir), str(Path(sys.executable).parent), os.environ["PATH"]])
    return {**os.environ, "PATH": path, "EM_PIPE_SKIP_ENV": "1", "SBATCH_LOG": str(log)}, log


def write_cfg(tmp_path, slurm_extra=None, **sections):
    slurm = {"mail_user": "someone@example.org", "preview": {"array": 4, "cpus": 2, "mem": "8G", "time": "01:00:00"},
             "render": {"array": 3, "partition": "ga100", "gres": "gpu:1"}, **(slurm_extra or {})}
    # The config is checked before anything is submitted: raw_dir must hold a file matching raw.file_pattern.
    (tmp_path / "raw").mkdir(exist_ok=True)
    (tmp_path / "raw" / "M10_D06_tile0-0.tif").touch()
    return synth.write_config(tmp_path / "cfg.yaml", tmp_path / "raw", tmp_path / "out", slurm=slurm,
                              render={"num_scales": 3}, **sections)


def run(cfg, env, *args, check=True):
    # From another directory with a relative config path: the script must resolve it before cd-ing.
    p = subprocess.run(["bash", str(ROOT / "run_pipeline.sh"), cfg.name, *args], cwd=cfg.parent, env=env,
                       capture_output=True, text=True)
    if check:
        assert p.returncode == 0, p.stderr
    return p


def calls(lines):
    """Each sbatch command line (fake sbatch log or --dry-run output) as {option: value}, 'script', 'args'."""
    out = []
    for line in lines:
        words = shlex.split(line)
        i = next(k for k, w in enumerate(words) if w.endswith(".sbatch"))
        opts = dict(w[2:].split("=", 1) if "=" in w else (w[2:], True) for w in words[:i] if w.startswith("--"))
        out.append({**opts, "script": words[i], "args": words[i + 1:]})
    return out


def names(stdout):
    return [c["job-name"] for c in calls(stdout.splitlines())]


def test_submits_the_chain(tmp_path, env):
    env, log = env
    cfg = write_cfg(tmp_path)
    p = run(cfg, env)
    jobs = calls(log.read_text().splitlines())
    assert [j["job-name"] for j in jobs] == CHAIN
    assert all(j["parsable"] is True for j in jobs)
    # Each job waits for the previous one (ids from the fake sbatch, cluster suffix stripped).
    assert "dependency" not in jobs[0]
    for k, j in enumerate(jobs[1:-1], start=1):
        assert j["dependency"] == f"afterok:{1000 + k}" and j["kill-on-invalid-dep"] == "yes"
    assert f"em-export: job {1000 + CHAIN.index('em-export') + 1}" in p.stdout
    # The summary is the last job: it waits for every job before it, however they end, and is given their list.
    last = jobs[-1]
    assert last["dependency"] == "afterany:" + ":".join(str(1000 + k) for k in range(1, len(CHAIN)))
    assert "kill-on-invalid-dep" not in last and last["script"] == "slurm/qc_summary.sbatch"
    chain = Path(last["args"][last["args"].index("--chain") + 1])
    assert chain.parent == tmp_path / "out" / "work" / "logs"
    assert [line.split("\t") for line in chain.read_text().splitlines()] == [
        [name, str(1000 + k)] for k, name in enumerate(CHAIN[:-1], start=1)]

    by_name = {j["job-name"]: j for j in jobs}
    cfg_abs = str(cfg.resolve())
    assert by_name["em-check"]["script"] == "slurm/check.sbatch" and by_name["em-check"]["args"] == [cfg_abs]
    assert by_name["em-align-solve"]["args"] == [cfg_abs, "solve"]
    assert by_name["em-pyramid-s2"]["args"] == [cfg_abs, "run", "--scale", "2"]
    run_ = by_name["em-preview-run"]
    assert (run_["array"], run_["cpus-per-task"], run_["mem"], run_["time"]) == ("0-3", "2", "8G", "01:00:00")
    assert run_["output"] == str(tmp_path / "out" / "work" / "logs" / "%x-%A_%a.out")
    assert by_name["em-preview-merge"]["output"] == str(tmp_path / "out" / "work" / "logs" / "%x-%j.out")
    assert "array" not in by_name["em-preview-merge"]
    assert by_name["em-render-run"]["array"] == "0-2" and by_name["em-render-run"]["gres"] == "gpu:1"
    assert by_name["em-render-init"]["mem"] == "16G" and "array" not in by_name["em-render-init"]
    assert by_name["em-align-run"]["array"] == "0-19"   # config.py default
    assert all(j["mail-user"] == "someone@example.org" for j in jobs)
    assert (tmp_path / "out" / "work" / "logs").is_dir()


def test_dry_run_prints_without_submitting(tmp_path, env):
    env, log = env
    cfg = write_cfg(tmp_path)
    p = run(cfg, env, "--dry-run")
    assert not log.exists() and not (tmp_path / "out").exists()
    lines = p.stdout.splitlines()
    assert len(lines) == len(CHAIN) and all(line.startswith("sbatch --parsable ") for line in lines)
    assert [shlex.split(line)[2] for line in lines] == [f"--job-name={name}" for name in CHAIN]
    assert "--dependency=afterok:<em-check>" in shlex.split(lines[1])
    assert any(w.startswith("--dependency=afterany:<em-check>:<em-preview-run>:") for w in shlex.split(lines[-1]))


def test_steps_from_to_and_summary(tmp_path, env):
    env, log = env
    cfg = write_cfg(tmp_path)
    jobs = calls(run(cfg, env, "--steps", "align,render", "--dry-run").stdout.splitlines())
    assert [j["job-name"] for j in jobs] == ["em-align-run", "em-align-solve", "em-render-init", "em-render-run"]
    assert "dependency" not in jobs[0] and jobs[2]["dependency"] == "afterok:<em-align-solve>"
    assert names(run(cfg, env, "--from", "intensity", "--dry-run").stdout) == CHAIN[CHAIN.index("em-intensity"):]
    assert names(run(cfg, env, "--to", "intensity", "--dry-run").stdout) == CHAIN[:CHAIN.index("em-intensity") + 1]
    assert names(run(cfg, env, "--from", "align", "--to", "intensity", "--dry-run").stdout) == [
        "em-align-run", "em-align-solve", "em-intensity"]
    assert "comes after" in run(cfg, env, "--from", "render", "--to", "align", "--dry-run", check=False).stderr
    # the summary alone: after the given jobs, and nothing waits for it
    jobs = calls(run(cfg, env, "--steps", "render,summary", "--after", "77", "--summary-after", "88:99",
                     "--dry-run").stdout.splitlines())
    assert [j["job-name"] for j in jobs] == ["em-render-init", "em-render-run", "em-qc-summary"]
    assert jobs[0]["dependency"] == "afterok:77"
    assert jobs[2]["dependency"] == "afterany:<em-render-init>:<em-render-run>:88:99"
    assert not log.exists()


def test_config_from_before_zcorrect_was_removed(tmp_path, env):
    """The slice-thickness step (zcorrect) is no longer part of the pipeline. A config written while it existed,
    with the step switched off and Slurm settings for its jobs, still submits the same chain, with notes that
    those entries have no effect; one that switches it on is refused before anything is submitted."""
    env, log = env
    old = {"zcorrect": {"array": 30, "time": "02:00:00"}, "zcorrect_solve": {"mem": "8G"}}
    cfg = write_cfg(tmp_path, zcorrect={"enabled": False}, slurm_extra=old)
    p = run(cfg, env, "--dry-run")
    assert names(p.stdout) == CHAIN and not any("zcorrect" in line for line in p.stdout.splitlines())
    notes = [line for line in p.stderr.splitlines() if line.startswith("note: ")]
    assert len(notes) == 3 and all("has no effect" in n and "dev/2026-11_zstep_correction/zcorrect_step" in n
                                   for n in notes)
    check_dataset_config(cfg)
    p = run(cfg, env, "--steps", "zcorrect", "--dry-run", check=False)
    assert p.returncode == 1 and "unknown step: zcorrect" in p.stderr and not p.stdout
    p = run(write_cfg(tmp_path, zcorrect={"enabled": True}), env, "--from", "intensity", "--dry-run", check=False)
    assert p.returncode == 1 and not p.stdout and "Traceback" not in p.stderr
    assert "zcorrect.enabled: true" in p.stderr and "removed from the pipeline on 26 Nov 2026" in p.stderr
    assert "dev/2026-11_zstep_correction/zcorrect_step" in p.stderr and "nothing was submitted" in p.stderr
    assert not log.exists()


def test_after_makes_the_first_job_wait(tmp_path, env):
    env, log = env
    jobs = calls(run(write_cfg(tmp_path), env, "--steps", "pyramid", "--after", "4242", "--dry-run").stdout.splitlines())
    assert jobs[0]["job-name"] == "em-pyramid-s1" and jobs[0]["dependency"] == "afterok:4242"
    assert jobs[1]["dependency"] == "afterok:<em-pyramid-s1>"


def test_bad_arguments(tmp_path, env):
    env, log = env
    cfg = write_cfg(tmp_path)
    assert "unknown step" in run(cfg, env, "--steps", "check,nope", check=False).stderr
    assert run(cfg, env, "--from", "align", "--steps", "check", check=False).returncode == 1
    assert run(cfg.with_name("missing.yaml"), env, check=False).returncode == 1
    p = run(cfg, env, "--steps", "upload", check=False)   # disabled: nothing to submit
    assert p.returncode == 1 and "upload skipped" in p.stderr
    assert not log.exists()


def test_config_cli(tmp_path, capsys):
    cfg = write_cfg(tmp_path)
    keys = ["upload.enabled", "render.num_scales", "render.blend_px", "nothing.here"]   # blend_px: a step default
    assert config._main(["--config", str(cfg), "--get", *keys, "--default", "x"]) == 0
    assert capsys.readouterr().out.splitlines() == ["false", "3", "256", "x"]
    assert config._main(["--config", str(cfg), "--sbatch-args", "check", "render"]) == 0
    lines = [line.split("\t") for line in capsys.readouterr().out.splitlines()]
    assert lines == [["check", f"--output={tmp_path / 'out' / 'work' / 'logs' / '%x-%j.out'}", "--mail-user=someone@example.org"],
                     ["render", f"--output={tmp_path / 'out' / 'work' / 'logs' / '%x-%A_%a.out'}", "--array=0-2",
                      "--partition=ga100", "--gres=gpu:1", "--mail-user=someone@example.org"]]
    assert config._main(["--config", str(cfg), "--sbatch-args"]) == 0
    assert [line.split("\t")[0] for line in capsys.readouterr().out.splitlines()] == list(config.JOBS)
    assert config._main(["--config", str(cfg), "--dump"]) == 0
    import json
    dump = json.loads(capsys.readouterr().out)
    assert dump["render"]["num_scales"] == 3 and dump["render"]["blend_px"] == 256 and dump["intensity"]["window"]
    assert "config_path" not in dump and dump["slurm"]["preview"]["mem"] == "8G" and "tissue_movie" in dump
    with pytest.raises(ValueError, match="quoted string"):
        config.sbatch_args({"output_dir": "/o", "slurm": {"align": {"time": 14400}}}, "align")
    with pytest.raises(ValueError, match="unknown keys"):
        config.sbatch_args({"output_dir": "/o", "slurm": {"align": {"memory": "4G"}}}, "align")


def test_example_config_is_the_defaults():
    """configs/example.yaml lists every key with its default value (and nothing else)."""
    example = config.load_config(ROOT / "configs" / "example.yaml")
    defaults = config.deep_merge(config.DEFAULTS, config.step_defaults())
    for key in ("name", "raw_dir", "output_dir", "config_path"):
        example.pop(key)
        defaults.pop(key, None)
    assert example == defaults


# Dataset configs live outside the repository: set EM_PIPE_CONFIGS to a folder of them to check those too.
DATASET_CONFIGS = [ROOT / "configs" / "example.yaml"] + sorted(
    Path(os.environ["EM_PIPE_CONFIGS"]).glob("*.yaml") if os.environ.get("EM_PIPE_CONFIGS") else [])


@pytest.mark.parametrize("path", DATASET_CONFIGS, ids=lambda p: p.name)
def test_dataset_configs_are_valid(path):
    check_dataset_config(path)


def check_dataset_config(path):
    from pipeline import movie_build, movie_tissue, preflight
    tools = {**movie_build.DEFAULTS, **movie_tissue.DEFAULTS}   # sections of the movie tools (not steps)
    cfg = config.load_config(path, step_defaults=config.step_defaults())
    # A step that was removed from the pipeline (zcorrect) may still be named, switched off, in a config written
    # while it existed, and its jobs may still have Slurm entries: neither has an effect. Switched on is an error.
    assert preflight.removed_entries(config._read_user(path))[0] == []
    assert set(cfg) <= set(config.DEFAULTS) | set(config.STEPS) | set(tools) | set(config.REMOVED_STEPS) | {"config_path"}
    for section in config.STEPS:   # no misspelt keys: every key is one the steps know
        assert set(cfg[section]) <= set(config.step_defaults(section)[section]), section
    for section in set(cfg) & set(tools):
        assert set(cfg[section]) <= set(tools[section]), section
    assert {k["action"] for k in cfg["check"]["known_issues"]} <= {"exclude", "ignore"}
    for job in config.JOBS:
        config.sbatch_args(cfg, job)   # raises on unknown keys or unquoted times


def test_config_base_is_merged_under_the_file(tmp_path):
    """A config with base: other.yaml gets other.yaml's settings, its own merged over them."""
    (tmp_path / "main.yaml").write_text("name: m\nraw_dir: /r\noutput_dir: /o\nrender: {downsample: 1, blend: cut}\n"
                                        "align: {local_windows: [[1, 2]]}\n")
    (tmp_path / "variant.yaml").write_text("base: main.yaml\noutput_dir: /o2\nrender: {downsample: 4}\n")
    cfg = config.load_config(tmp_path / "variant.yaml", step_defaults=config.step_defaults())
    assert cfg["output_dir"] == "/o2" and cfg["render"]["downsample"] == 4 and cfg["render"]["blend"] == "cut"
    assert cfg["align"]["local_windows"] == [[1, 2]] and "base" not in cfg
    (tmp_path / "main.yaml").write_text("base: variant.yaml\n")
    with pytest.raises(ValueError, match="loops"):
        config.load_config(tmp_path / "variant.yaml")


def test_precomputed_and_upload_steps_when_enabled(tmp_path, env):
    """After export: the precomputed export (init, array run, verify), then the upload (array run,
    finish), each only when enabled."""
    env, log = env
    cfg = write_cfg(tmp_path, precomputed={"enabled": True}, upload={"enabled": True, "destination": "gs://bucket/prefix"},
                    slurm_extra={"precomputed": {"array": 5}, "upload": {"array": 3, "time": "24:00:00"}})
    jobs = calls(run(cfg, env, "--from", "export", "--dry-run").stdout.splitlines())
    assert [j["job-name"] for j in jobs] == ["em-export", "em-quality-run", "em-quality-merge", "em-qc-summary",
                                             "em-precomputed-init", "em-precomputed-run",
                                             "em-precomputed-verify", "em-upload-run", "em-upload-finish"]
    jobs = [j for j in jobs if "quality" not in j["job-name"] and j["job-name"] != "em-qc-summary"]
    assert jobs[2]["array"] == "0-4" and jobs[2]["args"][1:] == ["run"] and jobs[4]["array"] == "0-2"
    assert jobs[5]["args"][1:] == ["finish"] and jobs[5]["dependency"] == "afterok:<em-upload-run>"
    assert jobs[1]["dependency"] == "afterok:<em-quality-merge>"     # not the summary: nothing waits for it
    p = run(write_cfg(tmp_path), env, "--from", "export", "--dry-run")
    assert names(p.stdout) == ["em-export", "em-quality-run", "em-quality-merge", "em-qc-summary"]
    assert "precomputed.enabled is false" in p.stderr


def test_shading_and_overlaps_steps_when_enabled(tmp_path, env):
    env, log = env
    cfg = write_cfg(tmp_path, shading={"enabled": True}, overlaps={"enabled": True},
                    slurm_extra={"overlaps": {"array": 7}})
    jobs = calls(run(cfg, env, "--from", "shading", "--dry-run").stdout.splitlines())
    assert [j["job-name"] for j in jobs][:4] == ["em-shading", "em-overlaps-run", "em-overlaps-merge", "em-intensity"]
    assert jobs[1]["array"] == "0-6" and jobs[1]["args"][1:] == ["run"] and jobs[2]["args"][1:] == ["merge"]
    p = run(write_cfg(tmp_path), env, "--steps", "shading,overlaps,intensity", "--dry-run")
    assert names(p.stdout) == ["em-intensity"]
    assert "shading.enabled is false" in p.stderr and "overlaps.enabled is false" in p.stderr


def test_patchgrid_step_when_enabled(tmp_path, env):
    """patchgrid: one job between the overlaps merge and intensity, only when enabled."""
    env, log = env
    cfg = write_cfg(tmp_path, overlaps={"enabled": True, "patch_grid": 8}, patchgrid={"enabled": True},
                    intensity={"balance_source": "overlaps"}, slurm_extra={"patchgrid": {"cpus": 4, "mem": "16G"}})
    jobs = calls(run(cfg, env, "--from", "overlaps", "--dry-run").stdout.splitlines())
    assert [j["job-name"] for j in jobs][:4] == ["em-overlaps-run", "em-overlaps-merge", "em-patchgrid", "em-intensity"]
    pg = jobs[2]
    assert pg["script"] == "slurm/patchgrid.sbatch" and pg["args"] == [str(cfg.resolve())] and "array" not in pg
    assert (pg["cpus-per-task"], pg["mem"]) == ("4", "16G") and pg["dependency"] == "afterok:<em-overlaps-merge>"
    assert jobs[3]["dependency"] == "afterok:<em-patchgrid>"
    assert names(run(cfg, env, "--steps", "patchgrid", "--dry-run").stdout) == ["em-patchgrid"]
    p = run(write_cfg(tmp_path, overlaps={"enabled": True}), env, "--from", "overlaps", "--dry-run")
    assert names(p.stdout)[:3] == ["em-overlaps-run", "em-overlaps-merge", "em-intensity"]
    assert "patchgrid.enabled is false: patchgrid skipped" in p.stderr


def test_config_is_checked_before_anything_is_submitted(tmp_path, env):
    """A misspelled key, settings that do not work together or a raw_dir without files: a message, no jobs."""
    env, log = env
    p = run(write_cfg(tmp_path, intensity={"balance_z_distnace": 1}), env, check=False)
    assert p.returncode == 1 and "did you mean 'balance_z_distance'" in p.stderr and "Traceback" not in p.stderr
    p = run(write_cfg(tmp_path, intensity={"balance_z_distance": 1}), env, "--dry-run", check=False)
    assert p.returncode == 1 and "needs balance_tiles: true and balance_source: overlaps" in p.stderr and not p.stdout
    cfg = write_cfg(tmp_path)
    (tmp_path / "raw" / "M10_D06_tile0-0.tif").unlink()
    p = run(cfg, env, check=False)
    assert p.returncode == 1 and "no file matching raw.file_pattern" in p.stderr
    assert not log.exists() and not (tmp_path / "out").exists()
