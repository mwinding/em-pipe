"""Shared command-line plumbing for pipeline steps."""

import argparse
import logging
import os
from pathlib import Path

from .config import load_config


def base_parser(description):
    """Parser with the options every step accepts."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", required=True, help="dataset YAML config")
    p.add_argument("--raw-dir", help="override raw_dir from the config")
    p.add_argument("--output-dir", help="override output_dir from the config")
    p.add_argument("--task-id", type=int, help="array task index (default: from Slurm, else 0)")
    p.add_argument("--num-tasks", type=int, help="number of array tasks (default: from Slurm, else 1)")
    p.add_argument("--overwrite", action="store_true", help="redo work whose output already exists")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def setup(args, step_defaults=None):
    """Configure logging and load the config from parsed ``base_parser`` args."""
    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return load_config(args.config, args.raw_dir, args.output_dir, step_defaults)


def task_info(args):
    """(task_id, num_tasks) from args, else the Slurm array environment, else (0, 1)."""
    if args.task_id is not None:
        return args.task_id, args.num_tasks or 1
    env = os.environ
    if "SLURM_ARRAY_TASK_ID" in env:
        tid = int(env["SLURM_ARRAY_TASK_ID"]) - int(env.get("SLURM_ARRAY_TASK_MIN", 0))
        n = int(env.get("SLURM_ARRAY_TASK_COUNT", 1))
        return tid, args.num_tasks or n
    return 0, args.num_tasks or 1


def chunks(items, size, overlap=0):
    """Split a sequence into consecutive chunks of ``size``.

    Each chunk is (core, extended): ``core`` is the items this chunk owns, ``extended``
    additionally includes the next ``overlap`` items (for pairwise work across chunk edges).
    """
    items = list(items)
    out = []
    for start in range(0, len(items), size):
        core = items[start:start + size]
        out.append((core, items[start:start + size + overlap]))
    return out


def my_chunks(all_chunks, task_id, num_tasks):
    """Strided assignment: task i takes chunks i, i+n, i+2n, ..."""
    return all_chunks[task_id::num_tasks]


def _create_tmp(directory, prefix, suffix):
    """Create an empty, uniquely named file the way ``open(..., "w")`` would.

    mkstemp makes files private (0600), and a chmod from the umask afterwards overrides the folder's
    default ACL (NEMO lab folders: rw for the lab, nothing for others). Creating with mode 0666 lets
    the OS apply the umask or the default ACL, so outputs get a normal write's permissions.
    """
    for _ in range(100):
        tmp = os.path.join(directory, f"{prefix}{os.urandom(4).hex()}{suffix}")
        try:
            os.close(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666))
            return tmp
        except FileExistsError:
            continue
    raise FileExistsError(f"no free temporary name {prefix}*{suffix} in {directory}")


def atomic_write(path, write_fn, suffix=None):
    """Call ``write_fn(tmp_path)`` then rename to ``path`` so readers never see partial files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the real extension last so numpy/matplotlib don't append or misdetect one.
    tmp = _create_tmp(path.parent, "." + path.name + ".", suffix or ".tmp" + path.suffix)
    try:
        write_fn(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
