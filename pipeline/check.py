"""Check the raw data: inventory, fault detection and the global slice list.

Scans ``raw_dir`` for the daily tile stacks, reads every header (cached by path, size and
mtime), checks each file and the files against each other, and writes ``check/files.csv``,
``slices.csv``, ``issues.csv`` and ``report.md`` (see docs/design.md). Exit code 1 if any
ERROR is not acknowledged in ``check.known_issues``.
"""

import hashlib
import json
import logging
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .cli import atomic_write, base_parser, setup
from .config import qc_path, step_dir, step_path
from .imagej_tiff import ImageJStack

log = logging.getLogger(__name__)

DEFAULTS = {
    "check": {
        # Expected voxel size in nm (number, or [x, y, z]); files off by >1% get a WARN. null: no check
        "voxel_size_nm": None,
        # TIME_GAP when the interval between slices exceeds gap_factor x the segment's median interval
        "gap_factor": 1.5,
        # Files modified more recently than this are still being written/copied: pending, not read
        "min_age_minutes": 60,
        # Bytes hashed from the middle of slices 0, n/2, n-1 to detect copied tiles (max half a slice)
        "hash_bytes": 1048576,
        # Parallel header reads (latency bound on network storage)
        "threads": 8,
        # Time zone of the label clock, e.g. Europe/London: label times are converted to UTC so slices
        # stay in order when the clocks go back (selection start/end stay local). null: as written
        "timezone": None,
        # Acknowledged problems: [{file, note, action: exclude|ignore}]; exclude drops the file's slices
        "known_issues": [],
    }
}

FILE_COLS = ["file", "month", "day", "part", "tile", "tile_row", "tile_col", "n_slices", "height", "width",
             "dtype", "byteorder", "data_offset", "contiguous", "size_bytes", "mtime", "voxel_x_nm",
             "voxel_y_nm", "voxel_z_nm", "first_timestamp", "last_timestamp", "sample_hashes", "status",
             "excluded", "exclude_reason"]
SLICE_COLS = ["z", "timestamp", "tile", "tile_row", "tile_col", "file", "index", "height", "width",
              "segment", "seam", "excluded", "exclude_reason", "label"]
ISSUE_COLS = ["severity", "code", "file", "timestamp", "z", "message", "known"]
SEVERITIES = ["ERROR", "WARN", "INFO"]
TIME_FMT = "%Y-%m-%dT%H:%M:%S"


# ----- reading -------------------------------------------------------------------------

def scan(raw_dir, file_re, recursive, skip_dir):
    """({relative path: (match, stat or None)} for files matching ``file_re``, [other files]).

    stat is None for a file that is listed but can't be opened. Hidden files and folders and ``skip_dir`` (the output folder) are skipped.
    """
    found, ignored = {}, []
    skip = os.path.realpath(skip_dir)

    def walk(d):
        for e in sorted(os.scandir(d), key=lambda e: e.name):
            if e.name.startswith("."):
                continue
            if e.is_dir():
                if recursive and os.path.realpath(e.path) != skip:
                    walk(e.path)
            elif e.is_file():
                rel = os.path.relpath(e.path, raw_dir)
                m = file_re.match(e.name)
                if m:
                    try:
                        found[rel] = (m, e.stat())
                    except OSError:
                        # Listed but can't be opened: on network shares, a file being replaced or copied.
                        found[rel] = (m, None)
                else:
                    ignored.append(rel)

    walk(raw_dir)
    return found, ignored


def read_header(path, hash_bytes):
    """Header fields, slice labels and sample hashes of one stack, as a JSON-serialisable dict."""
    st = ImageJStack(path)
    n_complete = (max(0, (st.size_bytes - st.data_offset) // st.slice_bytes) if st.truncated else st.n)
    length = min(int(hash_bytes), st.slice_bytes // 2)
    offset = (st.slice_bytes - length) // 2
    hashes, k = [], st.dtype.itemsize
    for i in (0, st.n // 2, st.n - 1):
        if st.contiguous:
            data = st.read_raw_bytes(i * st.slice_bytes + offset, length)
        else:
            data = st.read(i).tobytes()[offset:offset + length]
        # A sample cut short by truncation, or of one constant value (blank image), can't tell copies apart.
        useful = len(data) == length and data[k:] != data[:-k]
        hashes.append(hashlib.md5(data).hexdigest() if useful else "-")
    vx, vy, vz = st.voxel_size_nm
    return {"n_slices": st.n, "n_complete": int(n_complete), "height": st.height, "width": st.width,
            "dtype": st.dtype.name, "byteorder": st.byteorder, "data_offset": st.data_offset,
            "contiguous": st.contiguous, "voxel_x_nm": vx, "voxel_y_nm": vy, "voxel_z_nm": vz,
            "labels": st.labels, "sample_hashes": ";".join(hashes)}


def read_headers(raw_dir, stats, cache_dir, hash_bytes, threads, overwrite=False):
    """{file: (header or None, error or None)}, reusing cache entries whose (path, size, mtime) match."""
    n_read = []

    def one(rel):
        # "format": bump when read_header's output changes, so old cache entries are re-read.
        key = {"file": rel, "size": stats[rel].st_size, "mtime_ns": stats[rel].st_mtime_ns,
               "hash_bytes": int(hash_bytes), "format": 2}
        path = Path(cache_dir) / (rel.replace(os.sep, "__") + ".json")
        if not overwrite and path.exists():
            try:
                entry = json.loads(path.read_text())
                if entry["key"] == key:
                    return entry["header"], None
            except (ValueError, KeyError):
                pass  # corrupt cache entry: read the file again
        n_read.append(rel)
        try:
            header = read_header(Path(raw_dir) / rel, hash_bytes)
        except Exception as e:  # anything that stops us parsing the file makes it unreadable
            return None, f"{type(e).__name__}: {e}"
        # Failures are not cached so transient I/O errors are retried on the next run.
        atomic_write(path, lambda tmp: Path(tmp).write_text(json.dumps({"key": key, "header": header})))
        return header, None

    with ThreadPoolExecutor(max(1, int(threads))) as pool:
        out = dict(zip(stats, pool.map(one, stats)))
    log.info("read %d headers, %d from cache", len(n_read), len(stats) - len(n_read))
    return out


def parse_labels(labels, label_re, time_format, tz=None):
    """[(timestamp or None, tile or None)] for each slice label; times in naive UTC if ``tz`` is given."""
    out, prev = [], None
    for label in labels:
        m = label_re.match(label)
        ts = tile = None
        if m:
            g = m.groupdict()
            try:
                ts = datetime.strptime(f"{g['date']}_{g['time']}", time_format)
            except (KeyError, TypeError, ValueError):
                pass
            if ts is not None and tz is not None:
                ts = prev = _to_utc(ts, tz, prev)
            if g.get("row") is not None and g.get("col") is not None:
                tile = f"{int(g['row'])}-{int(g['col'])}"
        out.append((ts, tile))
    return out


def _to_utc(t, tz, prev):
    """Local time -> naive UTC. A time in the hour repeated when the clocks go back is the second pass
    if the previous slice of the file (``prev``, UTC) is already past the first."""
    first, second = (t.replace(tzinfo=tz, fold=f).astimezone(timezone.utc).replace(tzinfo=None)
                     for f in (0, 1))
    return second if prev is not None and first <= prev < second else first


# ----- checks --------------------------------------------------------------------------

def check_file(f, h, parsed, voxel_nm, add):
    """One file: TRUNCATED, DTYPE, VOXEL_SIZE, LABEL_COUNT, LABEL_PARSE, TILE_MISMATCH, NON_MONOTONIC."""
    rel, n = f["file"], h["n_slices"]
    timed = [(i, t) for i, (t, _) in enumerate(parsed) if t is not None]
    t0 = timed[0][1] if timed else None
    if h["n_complete"] < n:
        expected = h["data_offset"] + n * h["height"] * h["width"] * np.dtype(h["dtype"]).itemsize
        t = parsed[h["n_complete"]][0] if h["n_complete"] < len(parsed) else t0
        add("ERROR", "TRUNCATED", f"file is {expected - f['size_bytes']} bytes short: slices "
            f"{h['n_complete']}-{n - 1} of {n} are incomplete and excluded", rel, t)
    if h["dtype"] != "uint16":
        add("WARN", "DTYPE", f"pixel type {h['dtype']}, expected uint16", rel, t0)
    if voxel_nm is not None:
        expect = [float(v) for v in (voxel_nm if isinstance(voxel_nm, (list, tuple)) else [voxel_nm] * 3)]
        got = [h["voxel_x_nm"], h["voxel_y_nm"], h["voxel_z_nm"]]
        if any(g is not None and abs(g - e) > 0.01 * e for g, e in zip(got, expect)):
            shown = ", ".join("?" if g is None else f"{g:.3f}" for g in got)
            add("WARN", "VOXEL_SIZE", f"voxel size (x, y, z) = ({shown}) nm, expected {expect}", rel, t0)
    if len(parsed) != n:
        add("ERROR", "LABEL_COUNT", f"{len(parsed)} slice labels for {n} slices", rel, t0)
    bad = [label for label, (t, _) in zip(h["labels"], parsed) if t is None]
    if bad:
        add("ERROR", "LABEL_PARSE", f"{len(bad)} of {len(parsed)} labels don't match raw.label_pattern / "
            f"label_time_format, e.g. {bad[0]!r}", rel, t0)
    wrong = Counter(tile for _, tile in parsed if tile is not None and tile != f["tile"])
    if wrong:
        said = "filename says" if f.get("tile_from_name", True) else "the other labels say"
        add("ERROR", "TILE_MISMATCH", f"labels say tile {', '.join(wrong)} ({sum(wrong.values())} of "
            f"{len(parsed)}), {said} {f['tile']}", rel, t0)
    back = [(i, a, b) for (_, a), (i, b) in zip(timed, timed[1:]) if b <= a]
    if back:
        i, a, b = back[0]
        back_1h = timedelta(minutes=30) < a - b <= timedelta(hours=1)
        dst = " (clocks going back? set check.timezone)" if back_1h else ""
        add("ERROR", "NON_MONOTONIC", f"timestamps not strictly increasing at {len(back)} slice(s), first at "
            f"slice {i}: {a:%Y-%m-%d %H:%M:%S} then {b:%Y-%m-%d %H:%M:%S}{dst}", rel, b)


def duplicate_content(recs, headers, label_tile, add):
    """DUPLICATE_CONTENT between files of different tiles; returns {file: reason} of files to exclude."""
    by_hash = defaultdict(list)
    for rel, (h, _) in headers.items():
        if h is not None and h["sample_hashes"].strip("-;"):  # skip files with no usable sample
            by_hash[h["sample_hashes"]].append(rel)
    excluded = {}
    for group in by_hash.values():
        if len({recs[r]["tile"] for r in group}) < 2:
            continue
        # The copy is the file whose labels name another tile; if that doesn't single it out, drop all.
        copies = [r for r in group if label_tile.get(r) not in (None, recs[r]["tile"])]
        for r in copies if 0 < len(copies) < len(group) else group:
            others = ", ".join(f"{o} (tile {recs[o]['tile']})" for o in group if o != r)
            add("ERROR", "DUPLICATE_CONTENT", f"sampled pixel data identical to {others}; labels say tile "
                f"{label_tile.get(r)}; excluded", r, recs[r].get("first_timestamp"))
            excluded[r] = f"duplicate content of {others}"
    return excluded


def timestamp_mismatch(recs, parsed, skip, add):
    """Within each (month, day, part) every tile must have the same set of slice timestamps."""
    groups = defaultdict(dict)
    for rel, p in parsed.items():
        if rel not in skip:
            f = recs[rel]
            groups[(f["month"], f["day"], f["part"])][rel] = tuple(sorted({t for t, _ in p if t is not None}))
    for (month, day, part), seqs in groups.items():
        if len(set(seqs.values())) < 2:
            continue
        ref = Counter(seqs.values()).most_common(1)[0][0]
        for rel, seq in seqs.items():
            if seq != ref:
                lacking, extra = sorted(set(ref) - set(seq)), sorted(set(seq) - set(ref))
                add("ERROR", "TIMESTAMP_MISMATCH", f"{len(seq)} timestamps vs {len(ref)} in the other tiles "
                    f"of M{month:02d}_D{day:02d} part {part}: {len(lacking)} missing, {len(extra)} extra",
                    rel, min(lacking + extra))


def deduplicate(sl, add):
    """Keep one row per (timestamp, tile), preferring non-excluded rows; DUPLICATE_SLICE across files."""
    sl = sl.sort_values(["timestamp", "tile", "excluded", "file", "index"]).reset_index(drop=True)
    dup = sl.duplicated(["timestamp", "tile"])
    if dup.any():
        kept = sl[~dup].set_index(["timestamp", "tile"])["file"]
        pairs = defaultdict(list)
        for t, tile, rel in sl.loc[dup, ["timestamp", "tile", "file"]].itertuples(index=False):
            if kept[(t, tile)] != rel:  # repeats within one file are NON_MONOTONIC
                pairs[(rel, kept[(t, tile)], tile)].append(t)
        for (rel, other, tile), ts in pairs.items():
            add("ERROR", "DUPLICATE_SLICE", f"{len(ts)} slice(s) of tile {tile} also in {other}; using those",
                rel, ts[0])
    return sl[~dup].reset_index(drop=True)


def odd_shapes(sl, add):
    """TILE_SHAPE: tiles shaped unlike most tiles of the same slice are excluded (in place).

    Later steps rely on one tile shape per segment (e.g. preview stacks a tile's thumbnails).
    """
    n = sl.groupby(["timestamp", "height", "width"])["tile"].transform("size")
    odd = n < n.groupby(sl["timestamp"]).transform("max")
    for rel, d in sl[odd].groupby("file"):
        ref = sl[(sl["timestamp"] == d["timestamp"].iloc[0]) & ~odd].iloc[0]
        add("ERROR", "TILE_SHAPE", f"tile shape {d['height'].iloc[0]}x{d['width'].iloc[0]} differs from the "
            f"other tiles of its slices ({ref['height']}x{ref['width']}); {len(d)} slice(s) excluded",
            rel, d["timestamp"].iloc[0])
    sl.loc[odd & ~sl["excluded"], "exclude_reason"] = "tile shape differs from the other tiles"
    sl.loc[odd, "excluded"] = True


GRID = ["rows", "cols", "height", "width"]


def timeline(sl, recs, gap_factor, add):
    """Assign z, segment and seam to the slice rows (in place).

    Returns the per-z table (timestamp, grid rows/cols, tile height/width, usable, segment, interval).
    """
    times = pd.DatetimeIndex(sl["timestamp"].drop_duplicates().sort_values())
    sl["z"] = times.get_indexer(sl["timestamp"])
    g = sl.groupby("z")
    per = pd.DataFrame({"timestamp": times.to_numpy(), "rows": g["tile_row"].max() + 1,
                        "cols": g["tile_col"].max() + 1})
    shape = (sl.groupby(["z", "height", "width"]).size().rename("n").reset_index()
             .sort_values(["z", "n"], ascending=[True, False]).drop_duplicates("z").set_index("z"))
    per["height"], per["width"] = shape["height"], shape["width"]
    # A slice whose tiles are all excluded (a day still being copied, an excluded stray file) takes the
    # grid of the slice before it, so it doesn't split a segment.
    usable = per["usable"] = ~g["excluded"].all()
    if usable.any():
        per[GRID] = per[GRID].where(usable, axis=0).ffill().bfill().astype(int)
    key = per[GRID]
    per["segment"] = (key != key.shift()).any(axis=1).cumsum() - 1
    per["interval_s"] = per["timestamp"].diff().dt.total_seconds()
    inside = per["segment"] == per["segment"].shift()
    median = per.loc[inside, "interval_s"].groupby(per.loc[inside, "segment"]).median()
    per["median_s"] = per["segment"].map(median).fillna(per["interval_s"].median())
    per["gap"] = per["interval_s"] > gap_factor * per["median_s"]

    def ts(z):
        return per.at[z, "timestamp"]

    events = set()
    for z in per.index[per["gap"]]:
        add("WARN", "TIME_GAP", f"{per.at[z, 'interval_s'] / 60:.1f} min since the previous slice "
            f"({ts(z - 1):%Y-%m-%d %H:%M:%S}), {per.at[z, 'interval_s'] / per.at[z, 'median_s']:.1f}x the "
            f"median interval of {per.at[z, 'median_s']:.0f} s", "", ts(z))
        events.add(z)
    restarts = {}
    for rel, z in sl.groupby("file")["z"].min().items():
        f = recs[rel]
        if f["part"] > 1:
            k = (f["month"], f["day"], f["part"])
            restarts[k] = min(z, restarts.get(k, z))
    for (month, day, part), z in sorted(restarts.items(), key=lambda kv: kv[1]):
        add("INFO", "RESTART", f"acquisition restart: M{month:02d}_D{day:02d} part {part} starts", "", ts(z))
        events.add(z)
    for z in per.index[per["segment"].diff() > 0]:
        a, b = per.loc[z - 1], per.loc[z]
        add("INFO", "SEGMENT_CHANGE", f"segment {a.segment} -> {b.segment}: grid {a.rows}x{a.cols} -> "
            f"{b.rows}x{b.cols} (rows x cols), tile {a.height}x{a.width} -> {b.height}x{b.width} px (h x w)",
            "", ts(z))
        events.add(z)
    # A seam on a slice that is entirely excluded moves to the next usable slice, so later steps see it.
    seam, carry = np.zeros(len(per), bool), False
    for z in per.index:
        seam[z] = z in events or carry
        carry = seam[z] and not usable[z]
    sl["segment"] = per["segment"].to_numpy()[sl["z"]]
    sl["seam"] = seam[sl["z"]]
    return per


def _tile_name(rel, m, row, col):
    """``rel`` with the tile row/col in its filename replaced: the file that would hold that tile.

    ``"<rel> (tile r-c)"`` when the name has no tile (its tile came from the labels).
    """
    if m.groupdict().get("row") is None or m.groupdict().get("col") is None:
        return f"{rel} (tile {row}-{col})"
    name = os.path.basename(rel)
    for group, value in sorted((("row", row), ("col", col)), key=lambda gv: -m.start(gv[0])):
        name = name[:m.start(group)] + str(value) + name[m.end(group):]
    return os.path.join(os.path.dirname(rel), name)


def missing_tiles(sl, per, found, add):
    """MISSING_TILE, named by the file that would hold the tile:

    - a tile of a segment's full grid without a slice at some of its z (one issue per file and segment);
    - tiles dropped where the grid shrinks with the same tile shape (files not copied yet, more likely
      than a smaller imaged area), so an incomplete day can't pass as a new, smaller segment.

    Slices whose tiles are all excluded (e.g. a day still being copied) are not checked.
    """
    by_z = sl.groupby("z")
    present, sibling, usable = by_z["tile"].agg(set), by_z["file"].first(), per["usable"]

    def name(z, r, c):
        return _tile_name(sibling[z], found[sibling[z]][0], r, c)

    missing = defaultdict(list)
    for z, rows, cols, seg in zip(per.index, per["rows"], per["cols"], per["segment"]):
        if usable[z]:
            have = present[z]
            for r in range(rows):
                for c in range(cols):
                    if f"{r}-{c}" not in have:
                        missing[(name(z, r, c), f"{r}-{c}", seg)].append(z)
    for (file, tile, seg), zs in missing.items():
        add("ERROR", "MISSING_TILE", f"tile {tile} missing at {len(zs)} slice(s) of segment {seg}: "
            f"z {_ranges(zs)}", file, per.at[zs[0], "timestamp"])
    for z in per.index[per["segment"].diff() > 0]:
        a, b = per.loc[z - 1], per.loc[z]
        if usable[z] and (a.height, a.width) == (b.height, b.width) and b.rows <= a.rows and b.cols <= a.cols:
            for r, c in ((r, c) for r in range(a.rows) for c in range(a.cols) if r >= b.rows or c >= b.cols):
                add("ERROR", "MISSING_TILE", f"tile {r}-{c} absent from z {z} on: the grid shrinks from "
                    f"{a.rows}x{a.cols} to {b.rows}x{b.cols} with the same tile shape. If the imaged area "
                    f"really changed, list this file in check.known_issues (action: ignore)", name(z, r, c),
                    b.timestamp)


def _ranges(zs, limit=8):
    """'3-7, 9, 12-15' for a sorted list of ints."""
    runs, start = [], zs[0]
    for a, b in zip(zs, zs[1:] + [None]):
        if b != a + 1:
            runs.append(str(a) if start == a else f"{start}-{a}")
            start = b
    return ", ".join(runs[:limit]) + (f", ... ({len(runs)} ranges)" if len(runs) > limit else "")


# ----- main analysis -------------------------------------------------------------------

def _is_known(entry, rel):
    """Does a known_issues entry refer to this file (by relative path or by basename)?"""
    return bool(rel) and entry.get("file") in (rel, os.path.basename(rel))


def analyse(cfg, overwrite=False):
    """Run every check. Returns (files, slices, issues, per_z, ignored)."""
    c = cfg["check"]
    raw_dir = Path(cfg["raw_dir"])
    known_issues = c["known_issues"] or []
    for k in known_issues:
        if not (isinstance(k, dict) and k.get("file") and k.get("action", "ignore") in ("exclude", "ignore")):
            raise ValueError(f"check.known_issues: entries need a file and action exclude|ignore, got {k!r}")
    tz = ZoneInfo(c["timezone"]) if c.get("timezone") else None

    def known(rel):
        return next((k for k in known_issues if _is_known(k, rel)), None)

    found, ignored = scan(raw_dir, re.compile(cfg["raw"]["file_pattern"]), cfg["raw"]["recursive"],
                          cfg["output_dir"])
    issues = []

    def add(severity, code, message, file="", timestamp=None):
        issues.append({"severity": severity, "code": code, "file": file, "timestamp": timestamp,
                       "message": message})

    if not found:
        add("ERROR", "NO_FILES", f"no files in {raw_dir} match raw.file_pattern")
    cutoff = time.time() - 60 * float(c["min_age_minutes"])
    recs = {}
    for rel, (m, st) in found.items():
        g = m.groupdict()
        # A name without row/col (e.g. M06_D05_1.tif, a single-tile phase) takes its tile from its labels.
        named = g.get("row") is not None and g.get("col") is not None
        row, col = (int(g["row"]), int(g["col"])) if named else (None, None)
        recs[rel] = {"file": rel, "month": int(g["month"]), "day": int(g["day"]),
                     "part": int(g.get("part") or 1), "tile": f"{row}-{col}" if named else None,
                     "tile_row": row, "tile_col": col, "tile_from_name": named,
                     "size_bytes": st.st_size if st else None,
                     "mtime": datetime.fromtimestamp(st.st_mtime).strftime(TIME_FMT) if st else "",
                     "status": "pending" if st is None or st.st_mtime > cutoff else "ok"}
        if st is None:
            add("WARN", "UNAVAILABLE", "listed but can't be opened (being written or replaced?); "
                "treated as pending", rel)
    pending = {rel for rel, f in recs.items() if f["status"] == "pending"}
    log.info("%d files (%d pending), %d other files ignored", len(recs), len(pending), len(ignored))
    headers = read_headers(raw_dir, {rel: found[rel][1] for rel in recs if rel not in pending},
                           step_dir(cfg, "check", "cache"), c["hash_bytes"], c["threads"], overwrite)

    label_re, time_format = re.compile(cfg["raw"]["label_pattern"]), cfg["raw"]["label_time_format"]
    parsed, label_tile, label_errors = {}, {}, set()
    for rel, (h, err) in headers.items():
        f = recs[rel]
        if h is None:
            add("ERROR", "UNREADABLE", err, rel)
            continue
        f.update({k: v for k, v in h.items() if k in FILE_COLS})
        p = parsed[rel] = parse_labels(h["labels"], label_re, time_format, tz)
        times = [t for t, _ in p if t is not None]
        f["first_timestamp"], f["last_timestamp"] = (times[0], times[-1]) if times else (None, None)
        tiles = Counter(t for _, t in p if t is not None)
        label_tile[rel] = tiles.most_common(1)[0][0] if tiles else None
        if f["tile"] is None and label_tile[rel] is not None:
            f["tile"] = label_tile[rel]
            f["tile_row"], f["tile_col"] = map(int, f["tile"].split("-"))
        n_before = len(issues)
        check_file(f, h, p, c["voxel_size_nm"], add)
        if any(i["code"].startswith("LABEL_") for i in issues[n_before:]):
            label_errors.add(rel)
        if f["tile"] is None:
            add("ERROR", "NO_TILE", "neither the file name (raw.file_pattern row, col) nor the slice labels "
                "(raw.label_pattern row, col) give the tile; file not used", rel, f["first_timestamp"])
            del parsed[rel]
    timestamp_mismatch(recs, parsed, label_errors, add)

    # Whole-file exclusions: copied tiles, files of a day still being copied, known_issues 'exclude'.
    reasons = duplicate_content(recs, headers, label_tile, add)
    waiting = defaultdict(list)
    for rel in sorted(pending):
        f = recs[rel]
        waiting[(f["month"], f["day"], f["part"])].append(rel)
    for rel, f in recs.items():
        group = waiting.get((f["month"], f["day"], f["part"]))
        if group and rel not in pending:
            reasons.setdefault(rel, "waiting for pending " + ", ".join(group))
        k = known(rel)
        if k and k.get("action") == "exclude":
            reasons[rel] = k.get("note") or "excluded in check.known_issues"
    for rel, f in recs.items():
        f["excluded"], f["exclude_reason"] = rel in reasons, reasons.get(rel, "")

    rows = []
    for rel, p in parsed.items():
        f, h = recs[rel], headers[rel][0]
        for i, (t, _) in enumerate(p[:h["n_slices"]]):
            if t is not None:
                reason = reasons.get(rel) or ("truncated" if i >= h["n_complete"] else "")
                rows.append((t, f["tile"], f["tile_row"], f["tile_col"], rel, i, h["height"], h["width"],
                             bool(reason), reason, h["labels"][i]))
    sl = pd.DataFrame(rows, columns=["timestamp", "tile", "tile_row", "tile_col", "file", "index", "height",
                                     "width", "excluded", "exclude_reason", "label"])
    sl["timestamp"] = pd.to_datetime(sl["timestamp"])
    per = pd.DataFrame(columns=["timestamp", *GRID, "usable", "segment", "interval_s", "median_s", "gap"])
    if len(sl):
        sl = deduplicate(sl, add)
        odd_shapes(sl, add)
        per = timeline(sl, recs, float(c["gap_factor"]), add)
        missing_tiles(sl, per, found, add)
    else:
        sl = sl.assign(z=pd.Series(dtype=int), segment=pd.Series(dtype=int), seam=pd.Series(dtype=bool))

    for i in issues:
        i["known"] = known(i["file"]) is not None
    flagged = {i["file"] for i in issues if i["severity"] != "INFO"}
    for k in known_issues:
        if not any(_is_known(k, f) for f in flagged):
            add("INFO", "KNOWN_ISSUE_RESOLVED", f"listed in check.known_issues ({k.get('note', '')!r}, "
                f"{k.get('action', 'ignore')}) but has no ERROR or WARN now", k.get("file", ""))
            issues[-1]["known"] = True
    iss = pd.DataFrame(issues, columns=[col for col in ISSUE_COLS if col != "z"])
    iss["timestamp"] = pd.to_datetime(iss["timestamp"])
    z_of = pd.Series(per.index, index=pd.DatetimeIndex(per["timestamp"]), dtype=int)
    iss["z"] = iss["timestamp"].map(z_of).astype("Int64")
    iss["rank"] = iss["severity"].map(SEVERITIES.index)
    iss = iss.sort_values(["rank", "timestamp", "code", "file"], kind="stable", na_position="last")
    iss = iss[ISSUE_COLS].reset_index(drop=True)

    errors = set(iss.loc[iss["severity"] == "ERROR", "file"])
    for rel, f in recs.items():
        if f["status"] != "pending" and rel in errors:
            f["status"] = "error"
    files = pd.DataFrame(list(recs.values()), columns=FILE_COLS)
    return files, sl[SLICE_COLS].sort_values(["z", "tile"]).reset_index(drop=True), iss, per, ignored


# ----- output --------------------------------------------------------------------------

def _size(n):
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if n < 1000 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.2f} {unit}"
        n /= 1000


def _duration(td):
    hours = td / pd.Timedelta(hours=1)
    return f"{hours:.1f} h" if hours < 48 else f"{hours / 24:.1f} days"


def _t(ts, fmt="%Y-%m-%d %H:%M"):
    return "" if pd.isna(ts) else pd.Timestamp(ts).strftime(fmt)


def report(cfg, files, slices, issues, per, ignored):
    """report.md: summary, segments, issues by severity, time gaps, pending and ignored files."""
    L = [f"# Check report: {cfg.get('name') or Path(cfg['raw_dir']).name}", "",
         f"`{cfg['raw_dir']}`, checked {datetime.now():%Y-%m-%d %H:%M}.", ""]
    status = files["status"].value_counts()
    L.append(f"- **Files:** {len(files)} ({status.get('ok', 0)} ok, {status.get('error', 0)} with errors, "
             f"{status.get('pending', 0)} pending; {int(files['excluded'].sum())} excluded), "
             f"{_size(files['size_bytes'].sum())}")
    if len(slices):
        t0, t1 = slices["timestamp"].min(), slices["timestamp"].max()
        usable = slices.loc[~slices["excluded"], "z"].nunique()
        L.append(f"- **Slices:** {len(per)} (z 0-{len(per) - 1}, {usable} with usable tiles), "
                 f"{_t(t0)} to {_t(t1)} ({_duration(t1 - t0)}); {len(slices)} tile images, "
                 f"{int(slices['excluded'].sum())} excluded")
    sev = issues["severity"].value_counts()
    new = int(((issues["severity"] == "ERROR") & ~issues["known"]).sum())
    L += [f"- **Issues:** {sev.get('ERROR', 0)} ERROR ({new} not in check.known_issues), "
          f"{sev.get('WARN', 0)} WARN, {sev.get('INFO', 0)} INFO", ""]

    if len(per):
        L += ["## Segments", "", "| segment | grid (rows x cols) | tile (h x w) | z | time | slices |",
              "|---|---|---|---|---|---|"]
        for s, d in per.groupby("segment"):
            a = d.iloc[0]
            L.append(f"| {s} | {a.rows}x{a.cols} | {a.height}x{a.width} | {d.index[0]}-{d.index[-1]} | "
                     f"{_t(d.timestamp.iloc[0])} to {_t(d.timestamp.iloc[-1])} | {len(d)} |")
        L.append("")

    L += ["## Issues", ""]
    for severity in SEVERITIES:
        d = issues[(issues["severity"] == severity) & (issues["code"] != "TIME_GAP")]
        n_gaps = int(((issues["severity"] == severity) & (issues["code"] == "TIME_GAP")).sum())
        if not len(d) and not n_gaps:
            continue
        L += [f"### {severity} ({len(d) + n_gaps})", ""]
        for r in d.head(200).itertuples():
            where = " ".join(x for x in (f"`{r.file}`" if r.file else "",
                                         f"z={r.z}" if not pd.isna(r.z) else "") if x)
            L.append(f"- {'(known) ' if r.known else ''}**{r.code}** {where}: {r.message}")
        if len(d) > 200:
            L.append(f"- ... {len(d) - 200} more in issues.csv")
        if n_gaps:
            L.append(f"- **TIME_GAP** x {n_gaps}: see Time gaps")
        L.append("")

    gaps = per[per["gap"].astype(bool)] if len(per) else per
    if len(gaps):
        L += ["## Time gaps", "", "| z | from | to | minutes | x median |", "|---|---|---|---|---|"]
        for z, r in gaps.iterrows():
            L.append(f"| {z} | {_t(per.at[z - 1, 'timestamp'], '%Y-%m-%d %H:%M:%S')} | "
                     f"{_t(r.timestamp, '%Y-%m-%d %H:%M:%S')} | {r.interval_s / 60:.1f} | "
                     f"{r.interval_s / r.median_s:.1f} |")
        L.append("")
    pend = files[files["status"] == "pending"]
    if len(pend):
        L += [f"## Pending files ({len(pend)})", "",
              f"Modified less than {cfg['check']['min_age_minutes']} min ago; not read yet.", ""]
        L += [f"- `{r.file}` (modified {r.mtime})" for r in pend.itertuples()] + [""]
    if ignored:
        L += [f"## Ignored files ({len(ignored)})", "", "Not matching raw.file_pattern:", ""]
        L += [f"- `{f}`" for f in ignored[:20]]
        L += [f"- ... {len(ignored) - 20} more", ""] if len(ignored) > 20 else [""]
    return "\n".join(L)


def _write_csv(path, df):
    atomic_write(path, lambda tmp: df.to_csv(tmp, index=False))


def main(argv=None):
    p = base_parser("Inventory the raw files, detect faults and write the global slice list (check/).")
    args = p.parse_args(argv)
    cfg = setup(args, DEFAULTS)
    files, slices, issues, per, ignored = analyse(cfg, overwrite=args.overwrite)

    out = step_dir(cfg, "check")
    f = files.copy()
    for col in ("first_timestamp", "last_timestamp"):
        f[col] = pd.to_datetime(f[col]).dt.strftime(TIME_FMT)
    for col in ("tile_row", "tile_col", "n_slices", "height", "width", "data_offset"):
        f[col] = f[col].astype("Int64")
    _write_csv(out / "files.csv", f)
    _write_csv(out / "slices.csv", slices.assign(timestamp=slices["timestamp"].dt.strftime(TIME_FMT)))
    # Timestamps are UTC when check.timezone is set; later steps must interpret them the same way.
    atomic_write(out / "meta.json", lambda tmp: Path(tmp).write_text(
        json.dumps({"timezone": cfg["check"].get("timezone")})))
    _write_csv(out / "issues.csv", issues.assign(timestamp=issues["timestamp"].dt.strftime(TIME_FMT)))
    text = report(cfg, files, slices, issues, per, ignored)
    atomic_write(qc_path(cfg, "check_report.md"), lambda tmp: Path(tmp).write_text(text))

    new = issues[(issues["severity"] == "ERROR") & ~issues["known"]]
    log.info("%d slices, %d tile images; %d issues (%d unacknowledged errors); report: %s",
             len(per), len(slices), len(issues), len(new), out / "report.md")
    for r in new.head(20).itertuples():
        log.error("%s %s: %s", r.code, r.file, r.message)
    return 1 if len(new) else 0


if __name__ == "__main__":
    sys.exit(main())
