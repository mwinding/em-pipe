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
         "em-align-solve", "em-intensity", "em-render-init", "em-render-run", "em-pyramid-s1", "em-pyramid-s2"]
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


def write_cfg(tmp_path, **sections):
    slurm = {"mail_user": "someone@example.org", "preview": {"array": 4, "cpus": 2, "mem": "8G", "time": "01:00:00"},
             "render": {"array": 3, "partition": "ga100", "gres": "gpu:1"}}
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
    for k, j in enumerate(jobs[1:], start=1):
        assert j["dependency"] == f"afterok:{1000 + k}" and j["kill-on-invalid-dep"] == "yes"
    assert f"em-pyramid-s2: job {1000 + len(CHAIN)}" in p.stdout

    by_name = {j["job-name"]: j for j in jobs}
    cfg_abs = str(cfg.resolve())
    assert by_name["em-check"]["script"] == "slurm/check.sbatch" and by_name["em-check"]["args"] == [cfg_abs]
    assert by_name["em-align-solve"]["args"] == [cfg_abs, "solve"]
    assert by_name["em-pyramid-s2"]["args"] == [cfg_abs, "run", "--scale", "2"]
    run_ = by_name["em-preview-run"]
    assert (run_["array"], run_["cpus-per-task"], run_["mem"], run_["time"]) == ("0-3", "2", "8G", "01:00:00")
    assert run_["output"] == str(tmp_path / "out" / "logs" / "%x-%A_%a.out")
    assert by_name["em-preview-merge"]["output"] == str(tmp_path / "out" / "logs" / "%x-%j.out")
    assert "array" not in by_name["em-preview-merge"]
    assert by_name["em-render-run"]["array"] == "0-2" and by_name["em-render-run"]["gres"] == "gpu:1"
    assert by_name["em-render-init"]["mem"] == "16G" and "array" not in by_name["em-render-init"]
    assert by_name["em-align-run"]["array"] == "0-19"   # config.py default
    assert all(j["mail-user"] == "someone@example.org" for j in jobs)
    assert (tmp_path / "out" / "logs").is_dir()


def test_dry_run_prints_without_submitting(tmp_path, env):
    env, log = env
    cfg = write_cfg(tmp_path)
    p = run(cfg, env, "--dry-run")
    assert not log.exists() and not (tmp_path / "out").exists()
    lines = p.stdout.splitlines()
    assert len(lines) == len(CHAIN) and all(line.startswith("sbatch --parsable ") for line in lines)
    assert [shlex.split(line)[2] for line in lines] == [f"--job-name={name}" for name in CHAIN]
    assert "--dependency=afterok:<em-check>" in shlex.split(lines[1])


def test_steps_from_and_zcorrect(tmp_path, env):
    env, log = env
    cfg = write_cfg(tmp_path)
    jobs = calls(run(cfg, env, "--steps", "align,render", "--dry-run").stdout.splitlines())
    assert [j["job-name"] for j in jobs] == ["em-align-run", "em-align-solve", "em-render-init", "em-render-run"]
    assert "dependency" not in jobs[0] and jobs[2]["dependency"] == "afterok:<em-align-solve>"
    assert names(run(cfg, env, "--from", "intensity", "--dry-run").stdout) == CHAIN[CHAIN.index("em-intensity"):]
    cfg = write_cfg(tmp_path, zcorrect={"enabled": True})
    assert names(run(cfg, env, "--from", "intensity", "--dry-run").stdout)[:4] == [
        "em-intensity", "em-zcorrect-run", "em-zcorrect-solve", "em-render-init"]
    assert not log.exists()


def test_bad_arguments(tmp_path, env):
    env, log = env
    cfg = write_cfg(tmp_path)
    assert "unknown step" in run(cfg, env, "--steps", "check,nope", check=False).stderr
    assert run(cfg, env, "--from", "align", "--steps", "check", check=False).returncode == 1
    assert run(cfg.with_name("missing.yaml"), env, check=False).returncode == 1
    p = run(cfg, env, "--steps", "zcorrect", check=False)   # disabled: nothing to submit
    assert p.returncode == 1 and "zcorrect skipped" in p.stderr
    assert not log.exists()


def test_config_cli(tmp_path, capsys):
    cfg = write_cfg(tmp_path)
    keys = ["zcorrect.enabled", "render.num_scales", "render.blend_px", "nothing.here"]   # blend_px: a step default
    assert config._main(["--config", str(cfg), "--get", *keys, "--default", "x"]) == 0
    assert capsys.readouterr().out.splitlines() == ["false", "3", "256", "x"]
    assert config._main(["--config", str(cfg), "--sbatch-args", "check", "render"]) == 0
    lines = [line.split("\t") for line in capsys.readouterr().out.splitlines()]
    assert lines == [["check", f"--output={tmp_path / 'out' / 'logs' / '%x-%j.out'}", "--mail-user=someone@example.org"],
                     ["render", f"--output={tmp_path / 'out' / 'logs' / '%x-%A_%a.out'}", "--array=0-2",
                      "--partition=ga100", "--gres=gpu:1", "--mail-user=someone@example.org"]]
    assert config._main(["--config", str(cfg), "--sbatch-args"]) == 0
    assert [line.split("\t")[0] for line in capsys.readouterr().out.splitlines()] == list(config.JOBS)
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


@pytest.mark.parametrize("name", ["P667_EM05024_35h.yaml", "P667_test_2day.yaml"])
def test_dataset_configs_are_valid(name):
    cfg = config.load_config(ROOT / "configs" / name, step_defaults=config.step_defaults())
    assert set(cfg) <= set(config.DEFAULTS) | set(config.STEPS) | {"config_path"}
    for section in config.STEPS:   # no misspelt keys: every key is one the steps know
        assert set(cfg[section]) <= set(config.step_defaults(section)[section]), section
    assert {k["action"] for k in cfg["check"]["known_issues"]} == {"exclude"}
    for job in config.JOBS:
        config.sbatch_args(cfg, job)   # raises on unknown keys or unquoted times
