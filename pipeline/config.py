"""Dataset config: one YAML file per dataset.

Top-level defaults live here; each step module keeps its own section defaults in a
module-level ``DEFAULTS`` dict and passes it to ``load_config`` so step settings are
documented next to the code that uses them.
"""

import argparse
import copy
import importlib
import json
import sys
from pathlib import Path

import yaml

DEFAULTS = {
    "name": None,
    "raw_dir": None,
    "output_dir": None,
    "raw": {
        # Daily export files, e.g. M10_D06_tile1-1.tif or M09_D23_tile0-0_part2.tif
        "file_pattern": r"^M(?P<month>\d{2})_D(?P<day>\d{2})_tile(?P<row>\d+)-(?P<col>\d+)(?:_part(?P<part>\d+))?\.tif$",
        # Per-slice ImageJ labels, e.g. G460-0186_26-10-06_000146_0-1-1_InLens_raw.tif
        "label_pattern": r"^(?P<instrument>.+?)_(?P<date>\d{2}-\d{2}-\d{2})_(?P<time>\d{6})_(?P<group>\d+)-(?P<row>\d+)-(?P<col>\d+)_(?P<detector>[^_]+)_raw\.tif$",
        # Applied to f"{date}_{time}" from the label pattern
        "label_time_format": "%y-%m-%d_%H%M%S",
        "recursive": True,
    },
    # Restricts every step after check. Keys: start, end (timestamps), z_start, z_end (inclusive).
    "selection": {},
    # Slurm settings per job for run_pipeline.sh: array (number of tasks), cpus, mem, time (a quoted
    # string, e.g. "04:00:00"), partition, gres. Unset keys keep the #SBATCH defaults of slurm/<step>.sbatch.
    # Jobs: check, preview, preview_merge, stitch, stitch_merge, align, align_solve, intensity,
    # zcorrect, zcorrect_solve, render_init, render, pyramid (one job per scale).
    "slurm": {
        "mail_user": None,          # sbatch --mail-user (failure mails); null: Slurm's default
        "preview": {"array": 20},
        "stitch": {"array": 10},
        "align": {"array": 20},
        "zcorrect": {"array": 10},
        "render_init": {"mem": "16G"},   # render.sbatch's 160G is sized for render run
        "render": {"array": 20},
        "pyramid": {"array": 10},
    },
}

# Step modules, in pipeline order; each has a DEFAULTS section of the same name.
STEPS = ("check", "preview", "stitch", "align", "intensity", "destreak", "zcorrect", "render", "pyramid",
         "serve")
JOBS = ("check", "preview", "preview_merge", "stitch", "stitch_merge", "align", "align_solve", "intensity",
        "zcorrect", "zcorrect_solve", "render_init", "render", "pyramid")
ARRAY_JOBS = ("preview", "stitch", "align", "zcorrect", "render", "pyramid")
_SBATCH = {"cpus": "--cpus-per-task", "mem": "--mem", "time": "--time", "partition": "--partition",
           "gres": "--gres"}


def deep_merge(base, override):
    """Return a copy of ``base`` with ``override`` merged in recursively."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path, raw_dir=None, output_dir=None, step_defaults=None):
    """Load a dataset config, apply defaults and command-line path overrides.

    step_defaults: {section_name: defaults_dict} for the step(s) being run.
    """
    with open(path) as fh:
        user = yaml.safe_load(fh) or {}
    defaults = deep_merge(DEFAULTS, step_defaults or {})
    cfg = deep_merge(defaults, user)
    if raw_dir:
        cfg["raw_dir"] = raw_dir
    if output_dir:
        cfg["output_dir"] = output_dir
    for key in ("raw_dir", "output_dir"):
        if not cfg.get(key):
            raise ValueError(f"config {path}: '{key}' is not set")
        cfg[key] = str(Path(cfg[key]).expanduser())
    cfg["config_path"] = str(Path(path).resolve())
    return cfg


def step_dir(cfg, step, *parts):
    """Path to ``output_dir/<step>/<parts...>``, creating the step folder."""
    d = Path(cfg["output_dir"]) / step
    d.mkdir(parents=True, exist_ok=True)
    return d.joinpath(*parts) if parts else d


def get(cfg, dotted, default=None):
    """Look up a dotted key such as 'slurm.render.mem'."""
    node = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def step_defaults(*sections):
    """Merged DEFAULTS of the step modules owning ``sections`` (every step if none are given)."""
    out = {}
    for step in sections or STEPS:
        if step in STEPS:
            out = deep_merge(out, importlib.import_module(f"pipeline.{step}").DEFAULTS)
    return out


def sbatch_args(cfg, job):
    """sbatch options (job name aside) for one run_pipeline.sh job from the config's ``slurm`` section."""
    if job not in JOBS:
        raise ValueError(f"unknown job {job!r}; expected one of {JOBS}")
    slurm = cfg.get("slurm") or {}
    opts = slurm.get(job) or {}
    unknown = set(opts) - {"array", *_SBATCH}
    if unknown:
        raise ValueError(f"slurm.{job}: unknown keys {sorted(unknown)}")
    if not isinstance(opts.get("time", ""), str):
        # YAML reads an unquoted 04:00:00 as the integer 14400, which Slurm would take as minutes.
        raise ValueError(f"slurm.{job}.time must be a quoted string such as \"04:00:00\"")
    array = job in ARRAY_JOBS
    args = [f"--output={Path(cfg['output_dir']) / 'logs' / ('%x-%A_%a.out' if array else '%x-%j.out')}"]
    if array:
        args.append(f"--array=0-{int(opts.get('array', 1)) - 1}")
    args += [f"{flag}={opts[key]}" for key, flag in _SBATCH.items() if opts.get(key) is not None]
    if slurm.get("mail_user"):
        args.append(f"--mail-user={slurm['mail_user']}")
    return args


def _main(argv=None):
    """Config values and sbatch options for run_pipeline.sh (two calls cover a whole submission).

    python -m pipeline.config --config C --get output_dir zcorrect.enabled
        one value per line, step defaults included; JSON for non-strings (e.g. true, 7)
    python -m pipeline.config --config C --sbatch-args [JOB ...]
        one line per job (default: every job): the job, then its sbatch options, tab-separated
    """
    p = argparse.ArgumentParser(description=_main.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    what = p.add_mutually_exclusive_group(required=True)
    what.add_argument("--get", nargs="+", metavar="KEY", help="dotted keys, e.g. zcorrect.enabled")
    what.add_argument("--sbatch-args", nargs="*", metavar="JOB", choices=JOBS, help=f"jobs: {', '.join(JOBS)}")
    p.add_argument("--default", default="", help="--get: printed for keys that are not set")
    a = p.parse_args(argv)
    if a.sbatch_args is not None:
        cfg = load_config(a.config)
        for job in a.sbatch_args or JOBS:
            print("\t".join([job, *sbatch_args(cfg, job)]))
        return 0
    cfg = load_config(a.config, step_defaults=step_defaults(*{key.split(".")[0] for key in a.get}))
    for key in a.get:
        value = get(cfg, key, a.default)
        print("" if value is None else value if isinstance(value, str) else json.dumps(value))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
