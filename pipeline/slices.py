"""Access to check/slices.csv — the global slice list every later step works from."""

from pathlib import Path

import pandas as pd

from .imagej_tiff import ImageJStack


def slices_path(cfg):
    return Path(cfg["output_dir"]) / "check" / "slices.csv"


def load_slices(cfg, include_excluded=False, apply_selection=True):
    """check/slices.csv as a DataFrame, restricted by ``cfg['selection']``.

    Columns: z, timestamp (datetime64), tile, tile_row, tile_col, file, index, height,
    width, segment, seam, excluded, exclude_reason, label. Sorted by (z, tile).
    """
    path = slices_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run the check step first")
    df = pd.read_csv(path, dtype={"tile": str, "file": str, "exclude_reason": str, "label": str},
                     parse_dates=["timestamp"], keep_default_na=False)
    for col in ("seam", "excluded"):
        df[col] = df[col].astype(str).str.lower().isin(["true", "1"])
    if not include_excluded:
        df = df[~df["excluded"]]
    if apply_selection:
        df = select(df, cfg.get("selection") or {})
    return df.sort_values(["z", "tile"]).reset_index(drop=True)


def select(df, selection):
    """Apply a selection dict: start/end (timestamps, inclusive), z_start/z_end (inclusive)."""
    if selection.get("start"):
        df = df[df["timestamp"] >= pd.Timestamp(selection["start"])]
    if selection.get("end"):
        df = df[df["timestamp"] <= pd.Timestamp(selection["end"])]
    if selection.get("z_start") is not None:
        df = df[df["z"] >= int(selection["z_start"])]
    if selection.get("z_end") is not None:
        df = df[df["z"] <= int(selection["z_end"])]
    return df


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
