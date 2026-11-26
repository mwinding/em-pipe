"""Checks of a dataset config before anything is submitted: one function, ``validate``.

    python -m pipeline.preflight CONFIG [--no-raw] [--no-esb]
    python -m pipeline.config --config CONFIG --validate          (the same)

run_pipeline.sh runs it first and submits nothing when it reports an error. It checks, without tracebacks:

- the file reads as YAML and its ``base:`` chain resolves; raw_dir and output_dir are set;
- every key in every section is one em-pipe knows (a misspelled key is otherwise ignored silently and the
  default used instead), with the nearest known key as a suggestion;
- settings that only work together (the conditions the steps themselves raise, hours into a run): e.g.
  intensity.balance_z_distance > 0 needs overlaps.enabled, balance_source: overlaps and per_slice: false;
  patchgrid needs overlaps.patch_grid > 0; window: reference needs refmask and refstats;
- raw_dir exists and holds files matching raw.file_pattern;
- with decurtaining from a reference detector (destreak.enabled with reference_from / reference_to, or
  destreak.mode: reference): every raw file the run uses has its EsB file, with the same slice count and
  frame size (pipeline/esb.py; headers only). Raw files in subfolders count like any other.

Warnings do not stop a run; errors do.
"""

import argparse
import difflib
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from . import config as config_mod

SELECTION_KEYS = ("start", "end", "z_start", "z_end")
EXTRA_SECTIONS = ("movie_build", "movie_tissue")   # modules with a config section that are not pipeline steps


@dataclass
class Result:
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    esb: object = None            # pipeline.esb.Report when the EsB check ran

    @property
    def ok(self):
        return not self.errors

    def text(self, path=None):
        L = []
        for name, items in (("ERROR", self.errors), ("WARNING", self.warnings), ("note", self.notes)):
            L += [f"{name}: {m}" for m in items]
        head = f"config {path}: " if path else ""
        if self.errors:
            L.append(f"{head}{len(self.errors)} error{'s' if len(self.errors) != 1 else ''}: nothing was submitted")
        else:
            L.append(f"{head}OK" + (f" ({len(self.warnings)} warning{'s' if len(self.warnings) != 1 else ''})"
                                    if self.warnings else ""))
        return "\n".join(L)


def known_tree():
    """Every known key with its default: config.DEFAULTS merged with every module's section."""
    import importlib
    tree = config_mod.deep_merge(config_mod.DEFAULTS, config_mod.step_defaults())
    for name in EXTRA_SECTIONS:
        tree = config_mod.deep_merge(tree, importlib.import_module(f"pipeline.{name}").DEFAULTS)
    return tree


def _suggest(key, known):
    close = difflib.get_close_matches(str(key), [str(k) for k in known], n=1, cutoff=0.6)
    return f" (did you mean {close[0]!r}?)" if close else ""


def unknown_keys(user, tree=None):
    """Messages for keys of a config (after its base chain) that em-pipe does not know.

    A section is checked key by key, and so is every nested group whose defaults list its keys; a group
    without listed keys (e.g. stitch.segment_modes, whose keys are segment numbers) is free."""
    tree = tree if tree is not None else known_tree()
    out = []

    def walk(node, known, path):
        for key, value in node.items():
            here = f"{path}.{key}" if path else str(key)
            if key not in known:
                kind = "section" if not path else "key"
                out.append(f"unknown {kind} {here!r}{_suggest(key, known)}")
            elif isinstance(value, dict) and isinstance(known[key], dict) and known[key]:
                walk(value, known[key], here)

    top = {k: v for k, v in user.items() if k not in ("base", "slurm", "selection")}
    walk(top, tree, "")
    sel = user.get("selection") or {}
    if not isinstance(sel, dict):
        out.append("selection must be a group of keys (start, end, z_start, z_end)")
    else:
        out += [f"unknown key 'selection.{k}'{_suggest(k, SELECTION_KEYS)}" for k in sel if k not in SELECTION_KEYS]
    slurm = user.get("slurm") or {}
    if not isinstance(slurm, dict):
        out.append("slurm must be a group of jobs")
        return out
    job_keys = ("array", *config_mod._SBATCH)
    top_keys = (*config_mod.SLURM_KEYS, *config_mod.JOBS)
    for job, opts in slurm.items():
        if job in config_mod.SLURM_KEYS:
            continue
        if job not in config_mod.JOBS:
            out.append(f"unknown job 'slurm.{job}'{_suggest(job, top_keys)}")
        elif not isinstance(opts, dict):
            out.append(f"slurm.{job} must be a group of {', '.join(job_keys)}")
        else:
            out += [f"unknown key 'slurm.{job}.{k}'{_suggest(k, job_keys)}" for k in opts if k not in job_keys]
            if not isinstance(opts.get("time", ""), str):
                out.append(f"slurm.{job}.time must be a quoted string such as \"04:00:00\" (YAML reads an "
                           "unquoted time as a number)")
    return out


def combinations(cfg):
    """(errors, warnings) for settings that do not work together: what the steps would raise when they run."""
    from . import destreak, intensity, render
    E, W = [], []
    g = lambda key, default=None: config_mod.get(cfg, key, default)
    c = cfg["intensity"]

    def attempt(fn, *args):
        try:
            return fn(*args)
        except (ValueError, TypeError, KeyError) as e:
            E.append(str(e))

    overlaps, patchgrid = bool(g("overlaps.enabled")), bool(g("patchgrid.enabled"))
    if c.get("window") not in intensity.WINDOWS:
        E.append(f"intensity.window must be one of {intensity.WINDOWS}, not {c.get('window')!r}")
    if c.get("per_slice") not in (True, False, "slice"):
        E.append(f"intensity.per_slice must be true, false or slice, not {c.get('per_slice')!r}")
    if c.get("balance_source") not in ("thumbnails", "overlaps"):
        E.append(f"intensity.balance_source must be thumbnails or overlaps, not {c.get('balance_source')!r}")
    balanced = bool(c.get("balance_tiles"))
    if balanced and c.get("balance_source") == "overlaps" and not overlaps:
        E.append("intensity.balance_source: overlaps needs overlaps.enabled: true (the overlaps step measures the "
                 "relations it balances on)")
    zdist = attempt(intensity.z_distance, c)
    if zdist and not overlaps:
        E.append("intensity.balance_z_distance > 0 needs overlaps.enabled: true")
    if patchgrid:
        if not overlaps or not int(g("overlaps.patch_grid", 0) or 0) > 0:
            E.append("patchgrid.enabled needs overlaps.enabled: true and overlaps.patch_grid > 0 (the patch bins it "
                     "solves from)")
        if balanced and c.get("balance_source") != "overlaps":
            E.append("patchgrid.enabled needs intensity.balance_source: overlaps (render applies the patch shapes, "
                     "which the thumbnail balance does not know)")
    if c.get("window") == "reference":
        attempt(intensity.reference_settings, cfg)
        for step in ("refmask", "refstats"):
            if not g(f"{step}.enabled"):
                E.append(f"intensity.window: reference needs {step}.enabled: true (the {step} step)")
    elif c.get("window") == "percentile":
        if c.get("percentile_of") not in ("all", "tissue"):
            E.append(f"intensity.percentile_of must be all or tissue, not {c.get('percentile_of')!r}")
        else:
            for key in ("lo_percentile", "hi_percentile"):
                attempt(intensity.percentile_column, c[key], c["percentile_of"] == "tissue")
            if isinstance(c.get("lo_percentile"), (int, float)) and isinstance(c.get("hi_percentile"), (int, float)) \
                    and c["lo_percentile"] >= c["hi_percentile"]:
                E.append("intensity.lo_percentile must be below hi_percentile")
    if g("slurm.auto"):
        from . import resources
        problem = resources.check_facts(cfg.get("detected"))
        if problem:
            E.append(problem)
    if g("refstats.enabled") and not g("refmask.enabled"):
        E.append("refstats.enabled needs refmask.enabled: true (it measures inside the refmask step's mask)")
    if g("destreak.enabled"):
        p = attempt(destreak.settings, cfg)
        if p and p["reference_from"] is None:
            (W if p.get("mode") is None else []).append(
                "destreak.enabled without a reference (reference_from / reference_to): the filter-only path is "
                "used, which also removes the image's own structure along the stripes. Set destreak.mode: filter "
                "to state that choice, or give the reference detector's files")
    attempt(render._settings, cfg)
    if g("upload.enabled"):
        if g("precomputed.destination"):
            E.append("upload.enabled with precomputed.destination set: the export is written there directly, so "
                     "there is nothing to upload")
        elif not g("upload.destination"):
            E.append("upload.enabled needs upload.destination")
        if not g("precomputed.enabled"):
            E.append("upload.enabled needs precomputed.enabled: true (it uploads the precomputed export)")
    return E, W


def _writable_ancestor(path):
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    return p, os.access(p, os.W_OK)


def raw_files(cfg):
    """(relative paths of the files matching raw.file_pattern, files not matching), as check finds them."""
    from .check import scan
    found, ignored = scan(cfg["raw_dir"], re.compile(cfg["raw"]["file_pattern"]), cfg["raw"]["recursive"],
                          cfg["output_dir"])
    return sorted(found), ignored


def used_files(cfg, files):
    """The raw files the run uses: all of ``files`` without those check.known_issues excludes; once check
    has run (work/check/slices.csv), the files of the selected, non-excluded slices."""
    path = config_mod.step_path(cfg, "check", "slices.csv")
    if path.exists():
        try:
            from .slices import load_slices
            used = set(load_slices(cfg)["file"])
            if used:
                return sorted(f for f in files if f in used), "the selected slices of work/check/slices.csv"
        except Exception:   # an unreadable or older slice list: fall back to every file
            pass
    drop = [k.get("file") for k in (config_mod.get(cfg, "check.known_issues") or [])
            if isinstance(k, dict) and k.get("action") == "exclude"]
    return [f for f in files if f not in drop and os.path.basename(f) not in drop], "every raw file"


def validate(path, raw=True, esb=True):
    """Check the config at ``path``; returns a Result (``ok``, ``errors``, ``warnings``, ``notes``, ``esb``)."""
    res = Result()
    try:
        user = config_mod._read_user(path)
    except FileNotFoundError as e:
        res.errors.append(f"config file not found: {e.filename}")
        return res
    except (OSError, ValueError, yaml.YAMLError) as e:
        res.errors.append(f"the config cannot be read: {e}")
        return res
    if not isinstance(user, dict):
        res.errors.append("the config must be a YAML mapping of sections")
        return res
    tree = known_tree()
    res.errors += unknown_keys(user, tree)
    for key in ("raw", "check", "intensity", "render", "destreak", "slurm"):
        if user.get(key) is not None and not isinstance(user[key], dict):
            res.errors.append(f"{key} must be a group of settings, not {user[key]!r}")
            return res
    try:
        cfg = config_mod.load_config(path, step_defaults={k: v for k, v in tree.items()
                                                          if k not in config_mod.DEFAULTS})
    except ValueError as e:
        res.errors.append(str(e))
        return res
    try:
        file_re = re.compile(cfg["raw"]["file_pattern"])
        re.compile(cfg["raw"]["label_pattern"])
        if not {"month", "day"} <= set(file_re.groupindex):
            res.errors.append("raw.file_pattern needs the named groups month and day")
    except (re.error, TypeError) as e:
        res.errors.append(f"raw.file_pattern / raw.label_pattern is not a valid regular expression: {e}")
        return res
    E, W = combinations(cfg)
    res.errors += E
    res.warnings += W
    if not raw:
        return res
    raw_dir = Path(cfg["raw_dir"])
    if not raw_dir.is_dir():
        res.errors.append(f"raw_dir {raw_dir} " + ("is not a folder" if raw_dir.exists() else "does not exist"))
        return res
    try:
        files, ignored = raw_files(cfg)
    except OSError as e:
        res.errors.append(f"raw_dir {raw_dir} cannot be read: {e}")
        return res
    if not files:
        eg = f"; it holds e.g. {', '.join(ignored[:3])}" if ignored else "; it holds no files"
        res.errors.append(f"raw_dir {raw_dir} has no file matching raw.file_pattern{eg}")
        return res
    where, ok = _writable_ancestor(cfg["output_dir"])
    if not ok:
        res.errors.append(f"output_dir {cfg['output_dir']} cannot be written ({where} is not writable)")
    destreak_on = bool(config_mod.get(cfg, "destreak.enabled"))
    reference = config_mod.get(cfg, "destreak.reference_from") is not None
    if esb and destreak_on and reference and res.ok:
        from . import esb as esb_mod
        used, what = used_files(cfg, files)
        rep = esb_mod.check(raw_dir, used, cfg["destreak"]["reference_to"], cfg["destreak"]["reference_from"],
                            threads=config_mod.get(cfg, "check.threads", 8))
        res.esb = rep
        if rep.ok:
            res.notes.append(f"EsB check ({what}): {rep.summary()}")
        else:
            res.errors.append("\n".join(rep.lines() + [esb_mod.STOP_ADVICE]))
    elif destreak_on and not reference and res.ok:
        from . import esb as esb_mod
        n = len(esb_mod.find_candidates(raw_dir))
        if n:
            msg = (f"raw_dir holds {n} EsB files but destreak has no reference (reference_from / reference_to): "
                   "the curtains are filtered out, not swapped from EsB")
            (res.notes if config_mod.get(cfg, "destreak.mode") == "filter" else res.warnings).append(msg)
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config")
    p.add_argument("--no-raw", action="store_true", help="settings only: do not look at raw_dir")
    p.add_argument("--no-esb", action="store_true", help="skip the EsB file check")
    p.add_argument("--quiet", action="store_true", help="errors, warnings and notes only, on stderr")
    a = p.parse_args(argv)
    res = validate(a.config, raw=not a.no_raw, esb=not a.no_esb)
    if a.quiet:   # for run_pipeline.sh: stdout stays the job list; only what needs reading, on stderr
        lines = [f"{name}: {m}" for name, items in (("WARNING", res.warnings), ("note", res.notes)) for m in items]
        text = res.text(a.config) if not res.ok else "\n".join(lines)
        if text:
            print(text, file=sys.stderr)
    else:
        print(res.text(a.config), file=sys.stdout if res.ok else sys.stderr)
    return 0 if res.ok else 1


if __name__ == "__main__":
    sys.exit(main())
