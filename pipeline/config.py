"""Dataset config: one YAML file per dataset.

Top-level defaults live here; each step module keeps its own section defaults in a
module-level ``DEFAULTS`` dict and passes it to ``load_config`` so step settings are
documented next to the code that uses them.
"""

import argparse
import copy
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
}


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


def _main(argv=None):
    """`python -m pipeline.config --config C --get slurm.render.array` (used by run_pipeline.sh)."""
    p = argparse.ArgumentParser(description=_main.__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--get", required=True, help="dotted key")
    p.add_argument("--default", default="")
    a = p.parse_args(argv)
    with open(a.config) as fh:
        cfg = deep_merge(DEFAULTS, yaml.safe_load(fh) or {})
    value = get(cfg, a.get, a.default)
    print("" if value is None else value)


if __name__ == "__main__":
    sys.exit(_main())
