"""Slice-to-slice similarity per tile, to check the zcorrect step's slice spacing between independent areas.

``pipeline.zcorrect run`` keeps one NCC per slice pair: the median over its crops (one crop per tile). A change
in slice thickness is shared by the whole slice; a change in what a crop shows is not. This script measures the
same pairs with the same crops (zcorrect's own functions), but keeps every tile's NCC, so the spacing can be
solved per tile and compared (step_stats.py).

    python dev/2026-11_zstep_correction/scripts/tile_ncc.py --config CONFIG    (array job: one chunk set per task)

Writes ``<output_dir>/qc/zstep_correction/tile_ncc/chunk_<z0>-<z1>.npz``: z_a, z_b, tiles, ncc (pairs x tiles;
NaN where a tile has no crop). Existing chunk files are kept.
"""
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pipeline import zcorrect  # noqa: E402
from pipeline.cli import atomic_write, base_parser, chunks, my_chunks, setup, task_info  # noqa: E402
from pipeline.config import qc_path  # noqa: E402
from pipeline.slices import StackCache, load_slices, z_values  # noqa: E402

log = logging.getLogger("tile_ncc")


def main(argv=None):
    args = base_parser(__doc__.splitlines()[0]).parse_args(argv)
    cfg = setup(args, zcorrect.DEFAULTS)
    zc = cfg["zcorrect"]
    df = load_slices(cfg)
    zs = z_values(df)
    stitch, align = zcorrect._read_transforms(cfg)
    origins = zcorrect._crop_origins(df, stitch, align, zc["crop_px"])
    tiles = sorted({t for boxes in origins.values() for t in boxes})
    rows = {(r["z"], r["tile"]): r for r in df.to_dict("records")}
    segment = dict(zip(df["z"], df["segment"]))
    cache = StackCache(cfg["raw_dir"])
    out = qc_path(cfg, "zstep_correction") / "tile_ncc"
    out.mkdir(parents=True, exist_ok=True)
    task_id, num_tasks = task_info(args)
    for core, ext in my_chunks(chunks(zs, zc["chunk_slices"], zc["max_distance"]), task_id, num_tasks):
        path = out / f"chunk_{core[0]:06d}-{core[-1] + 1:06d}.npz"
        if path.exists():
            log.info("skip %s (exists)", path.name)
            continue
        cols = []
        for t in tiles:
            one = {s: {t: boxes[t]} for s, boxes in origins.items() if t in boxes}
            z_a, z_b, ncc = zcorrect._chunk_ncc(core, ext, rows, segment, stitch, align, one, cache, zc)
            cols.append(ncc)
        atomic_write(path, lambda p: np.savez(p, z_a=z_a, z_b=z_b, tiles=np.array(tiles), ncc=np.column_stack(cols)))
        log.info("%s: %d pairs x %d tiles", path.name, len(z_a), len(tiles))
    return 0


if __name__ == "__main__":
    sys.exit(main())
