"""The one-command run: ``em-pipe RAW_DIR OUTPUT_DIR`` (the script at the top of the repository).

Looks at the raw folder, writes OUTPUT_DIR/config.yaml on the standard preset (configs/presets/standard.yaml)
with this dataset's paths and what was detected in its files, checks it, and submits the whole job chain for
the plain volume and for the CLAHE volume. It stops before anything is submitted, with a message that names
what a person has to decide, when

- raw_dir holds no files matching the raw file pattern, or image files that do not match it (they would be
  left out; ``--skip-unmatched`` states that they are not part of the volume),
- EsB files are missing or do not match their raw files (decurtaining needs every one; ``--no-esb`` states
  that this run is without decurtaining),
- the curtain direction is conflicting or at an angle (``--direction`` states it),
- the written config fails the checks of pipeline/preflight.py,
- OUTPUT_DIR already holds another config.yaml.

    em-pipe RAW_DIR OUTPUT_DIR [--name N] [--no-clahe] [--no-esb] [--direction vertical|horizontal]
                               [--file-pattern REGEX] [--skip-unmatched] [--to STEP] [--dry-run]

--dry-run prints the config, the detected facts, the chosen Slurm resources and the sbatch commands, and
writes and submits nothing.

What stays a person's decision (never taken here): leaving out files or slices, the region the contrast
standard is measured in, a reference-tissue mask for a new stain, rotation, judging the result (the QC page
the chain ends with), uploads, and deleting anything.
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import yaml

from . import config as config_mod
from . import curtains, esb, preflight, resources
from .imagej_tiff import ImageJStack

ROOT = Path(__file__).resolve().parent.parent
PRESETS = ROOT / "configs" / "presets"
STOP = 2    # exit code: stopped for a person's decision (1: an error)

DECISIONS = """Decisions that stay with a person (nothing here takes them):
  - leaving out files or slices: check.known_issues, check.exclude_slices (nothing is excluded automatically;
    the check report and the slice-quality flags on the QC page say what to look at)
  - where the contrast standard is measured: refstats.region (default: all reference tissue of every slice)
  - a new stain: the reference-tissue mask was trained on our stain; look at qc/refmask.png and retrain
    (tools/refmask_train.py) if it marks the wrong tissue
  - rotation of the volume (render.rotate_deg), alignment windows for difficult slices (align.local_windows)
  - judging the result: the QC page and the volume in neuroglancer
  - uploads to a bucket, and deleting volumes or work folders"""


def _shape(path):
    try:
        return ImageJStack(path).shape
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def read_shapes(raw_dir, files, threads=8):
    """({file: (slices, height, width)}, {file: error}) from the TIFF headers."""
    with ThreadPoolExecutor(threads) as pool:
        got = dict(zip(files, pool.map(lambda f: _shape(Path(raw_dir) / f), files)))
    return ({f: s for f, s in got.items() if not isinstance(s, str)},
            {f: s for f, s in got.items() if isinstance(s, str)})


def _preset(name):
    path = Path(name) if os.sep in str(name) or str(name).endswith(".yaml") else PRESETS / f"{name}.yaml"
    if not path.is_file():
        have = ", ".join(sorted(p.stem for p in PRESETS.glob("*.yaml"))) or "none"
        raise SystemExit(f"em-pipe: preset {name!r} not found ({path}); presets: {have}")
    return path.resolve()


def _commit():
    try:
        r = subprocess.run(["git", "-C", str(ROOT), "describe", "--always", "--dirty", "--abbrev=10"],
                           capture_output=True, text=True, timeout=20)
        return r.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _q(s):
    return "'" + str(s).replace("'", "''") + "'"


def config_text(a, preset, name, facts, n_files, esb_rep, direction):
    """The dataset config as YAML text, with what was detected as comments."""
    L = [f"# Written by `em-pipe RAW_DIR OUTPUT_DIR` on {datetime.now():%Y-%m-%d %H:%M} (em-pipe {_commit()}).",
         "# The settings come from the preset (base:); below are this dataset's paths and what was found in its raw",
         "# files. To change something, edit this file and run  ./run_pipeline.sh <this file>  (--dry-run first).",
         f"base: {preset}", "", f"name: {name}", f"raw_dir: {os.path.abspath(a.raw_dir)}",
         f"output_dir: {os.path.abspath(a.output_dir)}", "",
         *([ "raw:", f"  file_pattern: {_q(a.file_pattern)}    # given with --file-pattern", ""] if a.file_pattern else []),
         f"# From the headers of the {n_files} raw files: the automatic Slurm resources are sized from these",
         "# (slurm.auto, pipeline/resources.py). tiles: the most tiles in one slice; tile_px: the largest tile.",
         "detected:"]
    L += [f"  {k}: {json.dumps(v)}" for k, v in facts.items()]
    L += ["", "destreak:"]
    if a.no_esb:
        L += ["  # --no-esb: this run is without decurtaining, by choice (no EsB check, no curtain direction).",
              "  enabled: false"]
    else:
        L += [f"  # EsB check: {esb_rep.summary()}, each with its raw file's slice count and frame size.",
              f"  reference_from: {_q(esb.REFERENCE_FROM)}", f"  reference_to: {_q(esb_rep.reference_to)}"]
        L += [f"  # {line}" for line in _wrap(direction["message"])]
        L += [f"  direction: {direction['use']}"]
    every = 32 if facts["slices"] <= 2000 else 64
    L += ["", "refstats:",
          f"  every: {every}    # the reference tissue is sampled every {every} slices (32 up to 2000 slices, else 64)", "",
          "render:", f"  name: {name}.ome.zarr", ""]
    return "\n".join(L)


def _wrap(text, width=104):
    import textwrap
    return textwrap.wrap(text, width)


def clahe_text(name):
    return "\n".join([
        "# The CLAHE volume: config.yaml with local contrast equalisation switched on (the preset's clip limit and",
        "# tile size). Same levels as the plain volume; its own volume, rendered from the raw tiles.",
        "#   ./run_pipeline.sh <this file> --steps render,pyramid,summary",
        "base: config.yaml", "", "render:", f"  name: {name}_clahe.ome.zarr", "  clahe:", "    enabled: true", ""])


def _write(path, text):
    """Write a config; an existing file (which says the same: checked before) is kept."""
    if path.exists():
        print(f"{path} exists and says the same: kept")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print(f"wrote {path}")


def _run_pipeline(cfg_path, extra, env):
    p = subprocess.run(["bash", str(ROOT / "run_pipeline.sh"), str(cfg_path), *extra], capture_output=True,
                       text=True, env=env)
    return p.returncode, p.stdout, p.stderr


def main(argv=None):
    p = argparse.ArgumentParser(prog="em-pipe", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("raw_dir", metavar="RAW_DIR", help="folder of the raw tile stacks (read only)")
    p.add_argument("output_dir", metavar="OUTPUT_DIR", help="folder for config.yaml, the volumes, qc/ and work/")
    p.add_argument("--name", help="dataset name: names the volumes (default: RAW_DIR's folder name)")
    p.add_argument("--no-clahe", action="store_true", help="only the plain volume")
    p.add_argument("--no-esb", action="store_true", help="run without decurtaining (stated in the config)")
    p.add_argument("--direction", choices=["vertical", "horizontal"],
                   help="curtain direction, as a person's statement (not measured)")
    p.add_argument("--file-pattern", metavar="REGEX",
                   help="raw.file_pattern for raw files named otherwise than M10_D06_tile1-1.tif (written to the config)")
    p.add_argument("--skip-unmatched", action="store_true",
                   help="image files that do not match the raw file pattern are not part of the volume (stated)")
    p.add_argument("--preset", default="standard", help="preset name in configs/presets, or a YAML path")
    p.add_argument("--to", metavar="STEP", help="submit the chain only up to and including this step (e.g. intensity, to "
                   "look at the levels before anything is rendered); continue with ./run_pipeline.sh CONFIG --from STEP")
    p.add_argument("--dry-run", action="store_true", help="show everything, write and submit nothing")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    say = lambda *x: print(*x, flush=True)
    # as typed, made absolute: links in the path are kept (a share is often reached through one)
    raw_dir, out_dir = Path(os.path.abspath(a.raw_dir)), Path(os.path.abspath(a.output_dir))
    if not raw_dir.is_dir():
        say(f"em-pipe: RAW_DIR {raw_dir} " + ("is not a folder" if raw_dir.exists() else "does not exist"))
        return 1
    preset = _preset(a.preset)
    base = config_mod.deep_merge(config_mod.DEFAULTS, config_mod._read_user(preset))
    name = a.name or raw_dir.name
    if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
        say(f"em-pipe: the name {name!r} would name the volume folder: letters, digits, . _ - only (--name)")
        return 1

    # 1. the raw files
    from .check import scan
    if a.file_pattern:
        try:
            ok = {"month", "day"} <= set(re.compile(a.file_pattern).groupindex)
        except re.error as e:
            say(f"em-pipe: --file-pattern is not a valid regular expression: {e}")
            return 1
        if not ok:
            say("em-pipe: --file-pattern needs the named groups month and day (see raw.file_pattern in configs/example.yaml)")
            return 1
        base["raw"]["file_pattern"] = a.file_pattern
    file_re = re.compile(base["raw"]["file_pattern"])
    found, ignored = scan(raw_dir, file_re, base["raw"]["recursive"], out_dir)
    files = sorted(found)
    if not files:
        eg = f"; it holds e.g. {', '.join(ignored[:3])}" if ignored else "; it holds no files"
        say(f"em-pipe: no file in {raw_dir} matches the raw file pattern (e.g. M10_D06_tile1-1.tif){eg}. For other "
            "names set raw.file_pattern in a config of your own (configs/example.yaml)")
        return 1
    # Image stacks that are not read are data left out: that is a person's decision, never a silent one.
    stop = []
    stray = [f for f in ignored if f.lower().endswith((".tif", ".tiff")) and not esb._TOKEN.search(os.path.basename(f))]
    if stray:
        say(f"{len(stray)} image files in {raw_dir} do not match the raw file pattern and would be left out, e.g. "
            f"{', '.join(stray[:4])}")
        if a.skip_unmatched:
            say("  left out, as stated with --skip-unmatched")
        else:
            say("  If they are not part of the volume (example images, joined copies of the per-day files), say so "
                "with --skip-unmatched. If they are, give the pattern that matches them with --file-pattern "
                "(raw.file_pattern, see configs/example.yaml).")
            stop.append("image files that would be left out")
    shapes, unreadable = read_shapes(raw_dir, files)
    if not shapes:
        say(f"em-pipe: none of the {len(files)} raw files has a readable TIFF header, e.g. {files[0]}: "
            f"{unreadable[files[0]]}")
        return 1
    groups = {f: tuple(int(found[f][0].groupdict().get(k) or 1) for k in ("month", "day", "part")) for f in shapes}
    facts = resources.facts(shapes, groups)
    raw_text = (f"{len(files)} files: {facts['slices']} slices, up to {facts['tiles']} tiles per slice, tiles up to "
                f"{facts['tile_px'][0]} x {facts['tile_px'][1]} px, {facts['tile_images']} tile images "
                f"({2 * facts['raw_gpx'] / 1000:.2f} TB at 16 bit)")
    say(f"raw data in {raw_dir}: {raw_text}")
    if unreadable:
        say(f"  {len(unreadable)} files have no readable header (the check step will report them), e.g. "
            f"{next(iter(unreadable))}")

    # 2. EsB, before anything else: every raw file needs its EsB file
    rep = direction = None
    if a.no_esb:
        say("decurtaining: off (--no-esb): stated in the config; no EsB check")
    else:
        rep = esb.check(raw_dir, files)
        say("\n".join(rep.lines()))
        if not rep.ok:
            say(esb.STOP_ADVICE)
            stop.append("EsB files missing or not matching")
    if stop:
        say(f"STOPPED ({'; '.join(stop)}): nothing was written or submitted.")
        return STOP
    if not a.no_esb:
        # 3. curtain direction
        if a.direction:
            direction = {"use": a.direction, "direction": a.direction,
                         "message": f"curtain direction {a.direction}: stated with --direction, not measured"}
        else:
            reference = lambda f: esb.reference_of(f, esb.REFERENCE_FROM, rep.reference_to)
            direction = curtains.detect(raw_dir, files, reference, configured=base.get("destreak", {}).get("direction", "vertical"))
        say(direction["message"])
        if direction["use"] is None:
            say("STOPPED: nothing was written or submitted.")
            return STOP

    # 4. the configs: checked in a temporary folder first, so nothing is written that would not run
    text, ctext = config_text(a, preset, name, facts, len(files), rep, direction), clahe_text(name)
    targets = [(out_dir / "config.yaml", text)] + ([] if a.no_clahe else [(out_dir / "config_clahe.yaml", ctext)])
    for path, t in targets:   # an existing config that says something else is never replaced
        if path.exists() and yaml.safe_load(path.read_text()) != yaml.safe_load(t):
            say(f"em-pipe: {path} exists and says something else: nothing was changed. Continue that run with  "
                f"./run_pipeline.sh {path}  or choose another OUTPUT_DIR")
            return 1
    tmp = tempfile.TemporaryDirectory()
    where = Path(tmp.name)
    for path, t in targets:
        (where / path.name).write_text(t)
    # 5. the checks every run starts with (the EsB files were checked above)
    for path, _ in targets:
        res = preflight.validate(where / path.name, esb=False)
        if not res.ok or res.warnings:
            say(res.text(path))
        if not res.ok:
            say("STOPPED: nothing was written or submitted.")
            return 1
    if a.dry_run:
        say(f"\n--- {out_dir / 'config.yaml'} (not written: --dry-run) ---\n{text}")
    else:
        for path, t in targets:
            _write(path, t)
        where = out_dir
        found_path = out_dir / "qc" / "em_pipe_detected.json"     # for the QC page: what was found before the run
        found_path.parent.mkdir(parents=True, exist_ok=True)
        found_path.write_text(json.dumps({
            "written": f"{datetime.now():%Y-%m-%d %H:%M}", "raw": raw_text,
            "esb": "not checked: this run is without decurtaining (--no-esb)" if rep is None else
                   f"{rep.summary()}; naming {rep.reference_to}",
            "direction": "" if direction is None else direction["message"],
            "direction_counts": None if direction is None else direction.get("counts")}, indent=1))
    cfg_path, clahe_path = where / "config.yaml", where / "config_clahe.yaml"
    cfg = config_mod.load_config(cfg_path, step_defaults=config_mod.step_defaults())
    say("\nSlurm resources per job (automatic, from the detected sizes; explicit slurm: settings win):")
    for job in config_mod.JOBS:
        opts = config_mod.job_options(cfg, job)
        if opts:
            say(f"  {job:20s} " + "  ".join(f"{k} {opts[k]}" for k in ("array", "cpus", "mem", "time") if k in opts))
    say("\n" + DECISIONS + "\n")

    # 6. submit: the whole chain, then the CLAHE volume once the levels exist
    env = dict(os.environ)     # the em-pipe script has set the environment up and says so (EM_PIPE_SKIP_ENV)
    shown = lambda s: s.replace(str(where), str(out_dir)) if a.dry_run else s
    dry = ["--dry-run"] if a.dry_run else []
    steps = config_mod.STEPS_IN_ORDER
    if a.to and a.to not in steps:
        say(f"em-pipe: --to {a.to}: unknown step (steps: {' '.join(steps)})")
        return 1
    code, out, err = _run_pipeline(cfg_path, (["--to", a.to] if a.to else []) + dry, env)
    say(shown(out).rstrip())
    if err.strip():
        say(shown(err).rstrip())
    if code:
        say("em-pipe: the chain was not submitted (see above)")
        return 1
    clahe = not a.no_clahe
    if clahe and a.to and steps.index(a.to) < steps.index("summary"):
        clahe = False
        say(f"the CLAHE volume was not submitted (--to {a.to}). When the plain chain is through: "
            f"./run_pipeline.sh {out_dir / 'config_clahe.yaml'} --steps render,pyramid,summary")
    if clahe:
        if a.dry_run:
            after = "<em-intensity>"
            summary_after = "<em-qc-summary>"
        else:
            ids = dict(re.findall(r"^(em-[\w-]+): job (\d+)", out, flags=re.M))
            after, summary_after = ids.get("em-intensity"), ids.get("em-qc-summary")
            if not after:
                say("em-pipe: the intensity job's id was not found in run_pipeline.sh's output: the CLAHE volume was "
                    f"not submitted. Submit it when the levels exist: ./run_pipeline.sh {clahe_path} --steps render,pyramid,summary")
                return 1
        extra = ["--steps", "render,pyramid,summary", "--after", after]
        if summary_after:
            extra += ["--summary-after", summary_after]
        code, out2, err2 = _run_pipeline(clahe_path, extra + dry, env)
        say(shown(out2).rstrip())
        if err2.strip():
            say(shown(err2).rstrip())
        if code:
            say("em-pipe: the CLAHE chain was not submitted (see above)")
            return 1
    if a.dry_run:
        say("\n--dry-run: nothing was written or submitted.")
    else:
        say(f"\nsubmitted. Follow with  squeue --me ; logs in {out_dir / 'work' / 'logs'}.\n"
            f"The chain ends with the QC page {out_dir / 'qc' / f'summary_{name}.html'}: it says whether every check "
            "passed, or where the chain stopped and why.")
    tmp.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
