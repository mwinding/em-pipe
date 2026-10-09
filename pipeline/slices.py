"""Access to check/slices.csv — the global slice list every later step works from."""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import step_path
from .imagej_tiff import ImageJStack

log = logging.getLogger(__name__)


def slices_path(cfg):
    return step_path(cfg, "check", "slices.csv")


def load_slices(cfg, include_excluded=False, apply_selection=True):
    """check/slices.csv as a DataFrame, restricted by ``cfg['selection']``.

    Columns: z, timestamp (datetime64), tile, tile_row, tile_col, file, index, height,
    width, segment, seam, excluded, exclude_reason, label. Sorted by (z, tile).
    """
    path = slices_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run the check step first")
    tz = (cfg.get("check") or {}).get("timezone")
    meta = path.with_name("meta.json")
    if meta.exists():
        written = json.loads(meta.read_text()).get("timezone")
        if written != tz:
            # Otherwise a date selection silently shifts by the UTC offset.
            raise ValueError(f"{path} was written with check.timezone {written!r} but this config has {tz!r}: "
                             "use the same value, or re-run check")
    df = pd.read_csv(path, dtype={"tile": str, "file": str, "exclude_reason": str, "label": str},
                     parse_dates=["timestamp"], keep_default_na=False)
    for col in ("seam", "excluded"):
        df[col] = df[col].astype(str).str.lower().isin(["true", "1"])
    if not include_excluded:
        df = df[~df["excluded"]]
    if apply_selection:
        df = select(df, cfg.get("selection") or {}, tz)
    return df.sort_values(["z", "tile"]).reset_index(drop=True)


def select(df, selection, timezone=None):
    """Apply a selection dict: start/end (timestamps, inclusive), z_start/z_end (inclusive).

    start/end are wall-clock times of the label clock. When ``timezone`` (check.timezone) is set,
    slices.csv holds UTC, so start/end are converted from that zone first.
    """
    def utc(value):
        t = pd.Timestamp(value)
        if timezone:
            # In the hour repeated when clocks go back, the first (summer-time) pass is meant.
            t = t.tz_localize(timezone, ambiguous=True, nonexistent="shift_forward")
            t = t.tz_convert("UTC").tz_localize(None)
        return t

    if selection.get("start"):
        df = df[df["timestamp"] >= utc(selection["start"])]
    if selection.get("end"):
        df = df[df["timestamp"] <= utc(selection["end"])]
    if selection.get("z_start") is not None:
        df = df[df["z"] >= int(selection["z_start"])]
    if selection.get("z_end") is not None:
        df = df[df["z"] <= int(selection["z_end"])]
    return df


def voxel_size_nm(cfg, files):
    """Raw (z, y, x) voxel size in nm: per axis the median over ``files`` in check/files.csv, else 8.

    render spaces its planes by this and zcorrect's positions are in units of it, so both use it.
    """
    path = step_path(cfg, "check", "files.csv")
    df = pd.read_csv(path, dtype={"file": str}) if path.exists() else pd.DataFrame({"file": []})
    df = df[df["file"].isin(set(files))]
    vox, unknown = [], []
    for col in ("voxel_z_nm", "voxel_y_nm", "voxel_x_nm"):
        v = pd.to_numeric(df[col], errors="coerce").median() if col in df else np.nan
        if not (np.isfinite(v) and v > 0):   # 0 nm would collapse the volume
            unknown.append(col)
            v = 8.0
        vox.append(float(v))
    if unknown:
        log.warning("no usable %s in %s for the selected files: assuming 8 nm", ", ".join(unknown), path)
    return vox


def z_values(df):
    """Sorted unique z in a slices frame."""
    return sorted(df["z"].unique().tolist())


class StackCache:
    """Keeps ImageJStack headers open by file so repeated slice reads don't re-parse headers."""

    def __init__(self, raw_dir):
        self.raw_dir = Path(raw_dir)
        self._stacks = {}

    def get(self, file):
        if file not in self._stacks:
            self._stacks[file] = ImageJStack(self.raw_dir / file)
        return self._stacks[file]

    def read(self, row, rows=slice(None), cols=slice(None)):
        """Read the slice referenced by a slices.csv row (needs 'file' and 'index')."""
        return self.get(row["file"]).read(int(row["index"]), rows, cols)
