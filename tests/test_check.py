"""Tests for pipeline.check on synthetic datasets with known faults."""

import os
import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest
import tifffile

import synth
from pipeline import check
from pipeline.config import load_config
from pipeline.imagej_tiff import ImageJStack
from pipeline.slices import load_slices


def config(tmp_path, raw_dir, output_dir=None, **check_cfg):
    sections = {k: check_cfg.pop(k) for k in ("raw", "selection") if k in check_cfg}
    return synth.write_config(tmp_path / "config.yaml", raw_dir, output_dir or tmp_path / "out",
                              check={"min_age_minutes": 0, **check_cfg}, **sections)


def run(cfg_path, *extra):
    return check.main(["--config", str(cfg_path), *extra])


def outputs(cfg_path):
    """(files, slices, issues, report) as written by the last run."""
    cfg = load_config(cfg_path)
    out = Path(cfg["output_dir"]) / "check"
    files = pd.read_csv(out / "files.csv", dtype={"tile": str}, keep_default_na=False)
    issues = pd.read_csv(out / "issues.csv", dtype={"file": str, "z": "Int64"})
    issues["file"] = issues["file"].fillna("")
    slices = load_slices(cfg, include_excluded=True, apply_selection=False)
    return files, slices, issues, (out / "report.md").read_text()


def pairs(issues, severity=None):
    d = issues if severity is None else issues[issues["severity"] == severity]
    return set(zip(d["code"], d["file"]))


def day_z(truth, prefix):
    return truth.files[f"{prefix}_tile0-0.tif"]


@pytest.fixture
def count_reads(monkeypatch):
    """Record the names of files whose headers are actually read (not taken from the cache)."""
    reads, real = [], check.read_header

    def read_header(path, hash_bytes):
        reads.append(Path(path).name)
        return real(path, hash_bytes)

    monkeypatch.setattr(check, "read_header", read_header)
    return reads


def test_clean_dataset_matches_truth(synth_2x2, tmp_path):
    t = synth_2x2
    # check ignores selection: the slice list always covers the whole acquisition
    cfg = config(tmp_path, t.raw_dir, voxel_size_nm=8, selection={"z_start": 5, "z_end": 9})
    assert run(cfg) == 0
    files, slices, issues, report = outputs(cfg)

    assert sorted(files["file"]) == sorted(t.files)
    assert (files["status"] == "ok").all() and not files["excluded"].any()
    assert set(files["tile"]) == set(t.tiles) and (files["part"] == 1).all()
    np.testing.assert_allclose(files[["voxel_x_nm", "voxel_y_nm", "voxel_z_nm"]].astype(float), 8.0)
    assert files["n_slices"].astype(int).sum() == len(t.timestamps) * len(t.tiles)

    assert len(slices) == len(t.timestamps) * len(t.tiles)
    assert slices.groupby("z")["tile"].apply(sorted).tolist() == [sorted(t.tiles)] * len(t.timestamps)
    per_z = slices.drop_duplicates("z").set_index("z")["timestamp"]
    assert per_z.index.tolist() == list(range(len(t.timestamps)))
    assert per_z.tolist() == [pd.Timestamp(ts) for ts in t.timestamps]
    for name, zs in t.files.items():
        rows = slices[slices["file"] == name].sort_values("index")
        assert rows["z"].tolist() == zs and rows["index"].tolist() == list(range(len(zs)))
        assert (rows["tile"] == name.split("_tile")[1][:3]).all()
    assert (slices["segment"] == 0).all() and not slices["seam"].any() and not slices["excluded"].any()
    assert (slices["height"] == t.tile_shape[0]).all() and (slices["width"] == t.tile_shape[1]).all()

    assert not pairs(issues, "ERROR") and not pairs(issues, "WARN")
    assert "| 0 | 2x2 | 192x224 | 0-23 |" in report


def test_scan_recursion_hidden_and_output_dir(tmp_path):
    t = synth.make_dataset(tmp_path / "raw")
    raw = t.raw_dir
    (raw / "day2").mkdir()
    day1 = sorted(n for n in t.files if n.startswith("M09_D24"))
    day2 = sorted(n for n in t.files if n.startswith("M09_D25"))
    for name in day2:
        (raw / name).rename(raw / "day2" / name)
    (raw / "notes.txt").write_text("not a stack")
    shutil.copy(raw / day1[0], raw / ("." + day1[0]))             # hidden
    (raw / "out").mkdir()
    shutil.copy(raw / day1[0], raw / "out" / "M09_D26_tile0-0.tif")  # inside output_dir

    cfg = config(tmp_path, raw, raw / "out")
    assert run(cfg) == 0
    files, slices, issues, report = outputs(cfg)
    assert sorted(files["file"]) == day1 + [f"day2/{n}" for n in day2]
    assert slices["z"].nunique() == len(t.timestamps)
    assert "## Ignored files (1)" in report and "notes.txt" in report

    cfg = config(tmp_path, raw, raw / "out", raw={"recursive": False})
    assert run(cfg) == 0
    files, slices, _, _ = outputs(cfg)
    assert sorted(files["file"]) == day1
    assert sorted(slices["z"].unique()) == day_z(t, "M09_D24")


def test_duplicate_tile_copy_like_d28(tmp_path):
    """Tile 1-0's file copied over the tile 0-1 name: its labels say 1-0 and its pixels equal 1-0's."""
    t = synth.make_dataset(tmp_path / "raw", faults={"duplicate": [("M09_D25", "0-1", "1-0")]})
    copy, src = "M09_D25_tile0-1.tif", "M09_D25_tile1-0.tif"
    cfg = config(tmp_path, t.raw_dir)
    assert run(cfg) == 1
    files, slices, issues, _ = outputs(cfg)

    assert pairs(issues, "ERROR") == {("DUPLICATE_CONTENT", copy), ("TILE_MISMATCH", copy)}
    assert src in issues.loc[issues["code"] == "DUPLICATE_CONTENT", "message"].iloc[0]
    f = files.set_index("file")
    assert f.at[copy, "excluded"] and f.at[copy, "status"] == "error"
    assert not f.at[src, "excluded"] and f.at[src, "status"] == "ok"
    excluded = slices[slices["excluded"]]
    assert set(zip(excluded["z"], excluded["tile"])) == {(z, "0-1") for z in day_z(t, "M09_D25")}
    assert (slices.groupby("z").size() == 4).all()  # the copy still fills its slot: no MISSING_TILE

    known = [{"file": copy, "note": "copy of tile 1-0", "action": "exclude"}]
    cfg = config(tmp_path, t.raw_dir, known_issues=known)
    assert run(cfg) == 0
    files, slices, issues, report = outputs(cfg)
    assert issues.loc[issues["file"] == copy, "known"].all() and len(issues[issues["file"] == copy]) == 2
    assert (slices.loc[slices["file"] == copy, "exclude_reason"] == "copy of tile 1-0").all()
    assert slices.loc[slices["file"] == copy, "excluded"].all()
    assert "(known) **DUPLICATE_CONTENT**" in report


def test_duplicate_undecidable_excludes_both(tmp_path):
    """Same pixels under two tile names with correct labels: we can't tell which is the copy."""
    t = synth.make_dataset(tmp_path / "raw", n_slices=6)
    a, b = "M09_D24_tile0-1.tif", "M09_D24_tile1-0.tif"
    with tifffile.TiffFile(t.raw_dir / a) as tif:
        labels = tif.imagej_metadata["Labels"]
    synth.write_imagej(t.raw_dir / a, tifffile.imread(t.raw_dir / b), labels)
    cfg = config(tmp_path, t.raw_dir)
    assert run(cfg) == 1
    files, slices, issues, _ = outputs(cfg)
    assert pairs(issues, "ERROR") == {("DUPLICATE_CONTENT", a), ("DUPLICATE_CONTENT", b)}
    assert files.set_index("file").loc[[a, b], "excluded"].all()
    assert set(slices.loc[slices["excluded"], "file"]) == {a, b}


def test_truncated_and_missing_tile_with_known_issues(tmp_path):
    # Cut into the last slice's pixels.
    t = synth.make_dataset(tmp_path / "raw", faults={"truncate": [("M09_D24", "1-1", 50000)],
                                                     "missing": [("M09_D25", "1-1")]})
    trunc, gone = "M09_D24_tile1-1.tif", "M09_D25_tile1-1.tif"
    d24, d25 = day_z(t, "M09_D24"), day_z(t, "M09_D25")
    cfg = config(tmp_path, t.raw_dir)
    assert run(cfg) == 1
    files, slices, issues, _ = outputs(cfg)

    assert pairs(issues, "ERROR") == {("TRUNCATED", trunc), ("MISSING_TILE", gone)}
    missing = issues[issues["code"] == "MISSING_TILE"].iloc[0]
    assert missing["z"] == d25[0] and f"z {d25[0]}-{d25[-1]}" in missing["message"]
    assert f"{len(d25)} slice(s)" in missing["message"]
    # only the incomplete last slice of the truncated file is excluded
    excluded = slices[slices["excluded"]]
    assert list(zip(excluded["z"], excluded["tile"], excluded["exclude_reason"])) == \
        [(d24[-1], "1-1", "truncated")]
    assert sorted(slices.loc[slices["tile"] == "1-1", "z"]) == d24
    assert gone not in set(files["file"])

    known = [{"file": trunc, "note": "copy interrupted", "action": "ignore"},
             {"file": gone, "note": "never exported", "action": "ignore"},
             {"file": "M09_D24_tile0-0.tif", "note": "old problem", "action": "ignore"}]
    cfg = config(tmp_path, t.raw_dir, known_issues=known)
    assert run(cfg) == 0
    files, slices, issues, _ = outputs(cfg)
    errors = issues[issues["severity"] == "ERROR"]
    assert len(errors) == 2 and errors["known"].all()
    assert pairs(issues, "INFO") == {("KNOWN_ISSUE_RESOLVED", "M09_D24_tile0-0.tif")}
    assert slices["excluded"].sum() == 1  # 'ignore' acknowledges but doesn't change exclusions


def test_header_and_label_faults(tmp_path):
    t = synth.make_dataset(tmp_path / "raw")
    raw = t.raw_dir

    def rewrite(name, data=lambda d: d, labels=lambda l: l, **kw):
        with tifffile.TiffFile(raw / name) as tif:
            d, l = tif.asarray(), list(tif.imagej_metadata["Labels"])
        synth.write_imagej(raw / name, data(d), labels(l), **kw)

    late = synth.LABEL.format(ts=t.timestamps[-1] + timedelta(seconds=30), r=0, c=0)
    rewrite("M09_D24_tile0-0.tif", labels=lambda l: l[::-1])
    rewrite("M09_D24_tile0-1.tif", labels=lambda l: l[:3] + ["garbage.tif"] + l[4:])
    rewrite("M09_D24_tile1-0.tif", data=lambda d: (d >> 8).astype(np.uint8), voxel_um=0.010)
    rewrite("M09_D24_tile1-1.tif", labels=lambda l: l[:-1])
    rewrite("M09_D25_tile0-0.tif", labels=lambda l: l[:-1] + [late])
    (raw / "M09_D26_tile0-0.tif").write_bytes(b"not a tiff" * 100)

    cfg = config(tmp_path, raw, voxel_size_nm=8)
    assert run(cfg) == 1
    files, slices, issues, _ = outputs(cfg)
    d24, d25, z_late = day_z(t, "M09_D24"), day_z(t, "M09_D25"), len(t.timestamps)
    # Slices without a usable label leave their tile missing; the late label adds a z with only tile 0-0.
    missing = {("M09_D24_tile0-1.tif", d24[3]), ("M09_D24_tile1-1.tif", d24[-1]), ("M09_D25_tile0-0.tif", d25[-1])}
    missing |= {(f"M09_D25_tile{tile}.tif", z_late) for tile in ("0-1", "1-0", "1-1")}
    expected = {("NON_MONOTONIC", "M09_D24_tile0-0.tif"), ("LABEL_PARSE", "M09_D24_tile0-1.tif"),
                ("LABEL_COUNT", "M09_D24_tile1-1.tif"), ("TIMESTAMP_MISMATCH", "M09_D25_tile0-0.tif"),
                ("UNREADABLE", "M09_D26_tile0-0.tif")} | {("MISSING_TILE", f) for f, _ in missing}
    assert pairs(issues, "ERROR") == expected
    m = issues[issues["code"] == "MISSING_TILE"]
    assert set(zip(m["file"], m["z"])) == missing
    assert pairs(issues, "WARN") == {("DTYPE", "M09_D24_tile1-0.tif"), ("VOXEL_SIZE", "M09_D24_tile1-0.tif")}
    assert "clocks going back" not in issues.loc[issues["code"] == "NON_MONOTONIC", "message"].iloc[0]
    assert files.set_index("file").at["M09_D26_tile0-0.tif", "status"] == "error"

    rev = slices[slices["file"] == "M09_D24_tile0-0.tif"].sort_values("index")
    assert rev["z"].tolist() == d24[::-1]  # labels, not file order, define z
    assert d24[3] not in set(slices.loc[slices["file"] == "M09_D24_tile0-1.tif", "z"])
    # the shifted timestamp adds one slice at the end of the timeline
    assert slices["z"].nunique() == len(t.timestamps) + 1
    assert slices.loc[slices["z"] == len(t.timestamps), "file"].tolist() == ["M09_D25_tile0-0.tif"]


def test_time_gap_and_restart(tmp_path):
    t = synth.make_dataset(tmp_path / "raw", start=datetime(2026, 9, 24, 10, 0), gaps={8: 1200}, parts=[16])
    cfg = config(tmp_path, t.raw_dir)
    assert run(cfg) == 0
    files, slices, issues, report = outputs(cfg)

    assert sorted(files["part"].unique()) == [1, 2]
    assert all(("_part2" in f) == (p == 2) for f, p in zip(files["file"], files["part"]))
    gaps = issues[issues["code"] == "TIME_GAP"]
    assert (gaps["severity"] == "WARN").all() and gaps["z"].tolist() == [8, 16]
    minutes = (t.timestamps[8] - t.timestamps[7]).total_seconds() / 60
    assert f"{minutes:.1f} min" in gaps["message"].iloc[0]
    restart = issues[issues["code"] == "RESTART"]
    assert restart["z"].tolist() == [16] and (restart["severity"] == "INFO").all()
    assert sorted(slices.loc[slices["seam"], "z"].unique()) == [8, 16]
    assert slices.loc[slices["seam"]].groupby("z").size().tolist() == [4, 4]
    assert "## Time gaps" in report and "| 8 |" in report

    cfg = config(tmp_path, t.raw_dir, gap_factor=4)  # the restart's 5 min pause is now below threshold
    assert run(cfg) == 0
    _, slices, issues, _ = outputs(cfg)
    assert issues.loc[issues["code"] == "TIME_GAP", "z"].tolist() == [8]
    assert sorted(slices.loc[slices["seam"], "z"].unique()) == [8, 16]


def test_segment_changes_3x3_to_2x2(tmp_path):
    """3x3 tiles, then smaller 3x3 tiles, then 2x2 larger tiles (as in the real data), as _partN files."""
    raw = tmp_path / "raw"
    raw.mkdir()
    specs = [((3, 3), (160, 176), 6, datetime(2026, 9, 18, 8, 0)),
             ((3, 3), (150, 170), 5, datetime(2026, 9, 18, 8, 30)),
             ((2, 2), (192, 224), 7, datetime(2026, 9, 18, 9, 0))]
    expected, z0 = [], 0
    for part, (grid, shape, n, start) in enumerate(specs, 1):
        t = synth.make_dataset(tmp_path / f"seg{part}", grid=grid, tile_shape=shape, n_slices=n, start=start,
                               seed=part)
        for name in t.files:
            (t.raw_dir / name).rename(raw / name.replace(".tif", f"_part{part}.tif"))
        expected.append((grid, shape, list(range(z0, z0 + n))))
        z0 += n

    cfg = config(tmp_path, raw)
    assert run(cfg) == 0
    files, slices, issues, report = outputs(cfg)
    assert len(files) == 9 + 9 + 4
    for seg, (grid, shape, zs) in enumerate(expected):
        rows = slices[slices["segment"] == seg]
        assert sorted(rows["z"].unique()) == zs
        assert set(rows["tile"]) == {f"{r}-{c}" for r in range(grid[0]) for c in range(grid[1])}
        assert len(rows) == len(zs) * grid[0] * grid[1]
        assert (rows["height"] == shape[0]).all() and (rows["width"] == shape[1]).all()
    change = issues[issues["code"] == "SEGMENT_CHANGE"]
    assert change["z"].tolist() == [6, 11] and (change["severity"] == "INFO").all()
    assert "grid 3x3 -> 2x2" in change["message"].iloc[1] and "150x170 -> 192x224" in change["message"].iloc[1]
    assert issues.loc[issues["code"] == "RESTART", "z"].tolist() == [6, 11]
    assert sorted(slices.loc[slices["seam"], "z"].unique()) == [6, 11]
    assert "MISSING_TILE" not in set(issues["code"])
    assert "| 2 | 2x2 | 192x224 | 11-17 |" in report


def test_cache_reuse(tmp_path, count_reads):
    t = synth.make_dataset(tmp_path / "raw", n_slices=6)
    cfg = config(tmp_path, t.raw_dir)
    assert run(cfg) == 0 and sorted(count_reads) == sorted(t.files)
    first = outputs(cfg)

    count_reads.clear()
    assert run(cfg) == 0 and count_reads == []
    second = outputs(cfg)
    for a, b in zip(first[:3], second[:3]):
        pd.testing.assert_frame_equal(a, b)

    changed = t.raw_dir / "M09_D24_tile1-1.tif"
    st = changed.stat()
    os.utime(changed, ns=(st.st_atime_ns, st.st_mtime_ns - 10**9))
    assert run(cfg) == 0 and count_reads == [changed.name]

    count_reads.clear()
    assert run(cfg, "--overwrite") == 0 and len(count_reads) == len(t.files)


def test_pending_file_not_read(tmp_path, count_reads):
    """A day still arriving: tile 0-1 being copied, tiles 1-0 and 1-1 not started yet."""
    t = synth.make_dataset(tmp_path / "raw")
    old = time.time() - 2 * 3600
    for name in t.files:
        os.utime(t.raw_dir / name, (old, old))
    fresh = "M09_D25_tile0-1.tif"
    os.utime(t.raw_dir / fresh)  # still being copied
    for name in ("M09_D25_tile1-0.tif", "M09_D25_tile1-1.tif"):
        (t.raw_dir / name).unlink()
    cfg = config(tmp_path, t.raw_dir, min_age_minutes=60)
    assert run(cfg) == 0
    files, slices, issues, report = outputs(cfg)

    assert fresh not in count_reads and len(count_reads) == len(t.files) - 3
    assert files.set_index("file").at[fresh, "status"] == "pending"
    assert not pairs(issues, "ERROR") and not pairs(issues, "INFO")  # no MISSING_TILE, no SEGMENT_CHANGE
    assert sorted(slices["z"].unique()) == list(range(len(t.timestamps)))  # z already assigned
    d25 = slices["z"].isin(day_z(t, "M09_D25"))
    assert slices.loc[d25, "excluded"].all() and not slices.loc[~d25, "excluded"].any()
    assert slices.loc[d25, "exclude_reason"].str.contains("pending").all()
    assert (slices["segment"] == 0).all() and not slices["seam"].any()
    assert "## Pending files (1)" in report and fresh in report


def test_listed_but_unopenable_file_is_pending(tmp_path, monkeypatch):
    """Seen on the real share: a file being replaced is listed but stat() fails. Don't crash."""
    t = synth.make_dataset(tmp_path / "raw")
    gone = "M09_D25_tile1-1.tif"
    real_scandir = os.scandir

    class Entry:
        def __init__(self, e):
            self._e = e

        def __getattr__(self, name):
            return getattr(self._e, name)

        def stat(self, **kw):
            if self._e.name == gone:
                raise FileNotFoundError(2, "No such file or directory", self._e.path)
            return self._e.stat(**kw)

    monkeypatch.setattr(check.os, "scandir", lambda d: [Entry(e) for e in real_scandir(d)])
    cfg = config(tmp_path, t.raw_dir)
    assert run(cfg) == 0
    files, slices, issues, report = outputs(cfg)
    assert files.set_index("file").at[gone, "status"] == "pending"
    assert ("UNAVAILABLE", gone) in pairs(issues, "WARN")
    d25 = slices["z"].isin(day_z(t, "M09_D25"))
    assert slices.loc[d25, "excluded"].all() and not slices.loc[~d25, "excluded"].any()


def test_same_tile_in_two_folders(tmp_path):
    t = synth.make_dataset(tmp_path / "raw", n_slices=6)
    (t.raw_dir / "copy").mkdir()
    shutil.copy(t.raw_dir / "M09_D24_tile0-0.tif", t.raw_dir / "copy" / "M09_D24_tile0-0.tif")
    cfg = config(tmp_path, t.raw_dir)
    assert run(cfg) == 1
    files, slices, issues, _ = outputs(cfg)
    assert pairs(issues, "ERROR") == {("DUPLICATE_SLICE", "copy/M09_D24_tile0-0.tif")}
    assert len(slices) == 6 * 4 and not slices.duplicated(["z", "tile"]).any()
    assert set(slices.loc[slices["tile"] == "0-0", "file"]) == {"M09_D24_tile0-0.tif"}


def test_empty_and_all_pending(tmp_path):
    (tmp_path / "empty").mkdir()
    cfg = config(tmp_path, tmp_path / "empty")
    assert run(cfg) == 1
    files, slices, issues, _ = outputs(cfg)
    assert pairs(issues) == {("NO_FILES", "")} and len(files) == 0 and len(slices) == 0

    t = synth.make_dataset(tmp_path / "raw", n_slices=4)  # just written, so all pending
    cfg = config(tmp_path, t.raw_dir, min_age_minutes=60)
    assert run(cfg) == 0
    files, slices, issues, report = outputs(cfg)
    assert (files["status"] == "pending").all() and len(slices) == 0 and not pairs(issues, "ERROR")
    assert f"## Pending files ({len(t.files)})" in report


def test_grid_shrink_reports_missing_tiles(tmp_path):
    """A day without its whole second tile row (e.g. an rsync that hasn't got there yet) must fail check,
    not pass as a new 1x2 segment."""
    t = synth.make_dataset(tmp_path / "raw", faults={"missing": [("M09_D25", "1-0"), ("M09_D25", "1-1")]})
    gone = ["M09_D25_tile1-0.tif", "M09_D25_tile1-1.tif"]
    d25 = day_z(t, "M09_D25")
    cfg = config(tmp_path, t.raw_dir)
    assert run(cfg) == 1
    _, slices, issues, _ = outputs(cfg)
    m = issues[issues["severity"] == "ERROR"]
    assert set(zip(m["code"], m["file"], m["z"])) == {("MISSING_TILE", f, d25[0]) for f in gone}
    assert sorted(slices.loc[slices["seam"], "z"].unique()) == [d25[0]]  # still a segment change

    cfg = config(tmp_path, t.raw_dir, known_issues=[{"file": f, "note": "smaller area", "action": "ignore"}
                                                    for f in gone])
    assert run(cfg) == 0


def test_tile_shape_mismatch_excluded(tmp_path):
    t = synth.make_dataset(tmp_path / "raw", n_slices=6)
    odd = "M09_D24_tile1-1.tif"
    with tifffile.TiffFile(t.raw_dir / odd) as tif:
        data, labels = tif.asarray(), tif.imagej_metadata["Labels"]
    synth.write_imagej(t.raw_dir / odd, data[:, :-10], labels)
    cfg = config(tmp_path, t.raw_dir)
    assert run(cfg) == 1
    _, slices, issues, _ = outputs(cfg)
    assert pairs(issues, "ERROR") == {("TILE_SHAPE", odd)} and not pairs(issues, "INFO")
    assert "182x224" in issues["message"].iloc[0] and "192x224" in issues["message"].iloc[0]
    excluded = slices[slices["excluded"]]
    assert set(excluded["file"]) == {odd} and len(excluded) == 6
    usable = slices[~slices["excluded"]]
    assert (usable["height"] == 192).all() and (slices["segment"] == 0).all()


def test_blank_and_header_only_files_are_not_copies(tmp_path):
    """Constant images, or files cut off before the sampled bytes, hash alike without being copies."""
    t = synth.make_dataset(tmp_path / "raw", n_slices=6, start=datetime(2026, 9, 24, 10, 0))
    for name in ("M09_D24_tile0-0.tif", "M09_D24_tile0-1.tif"):
        with tifffile.TiffFile(t.raw_dir / name) as tif:
            data, labels = tif.asarray(), tif.imagej_metadata["Labels"]
        synth.write_imagej(t.raw_dir / name, np.full_like(data, 1000), labels)
    for name in ("M09_D24_tile1-0.tif", "M09_D24_tile1-1.tif"):
        offset = ImageJStack(t.raw_dir / name).data_offset
        with open(t.raw_dir / name, "r+b") as fh:
            fh.truncate(offset + 10)
    cfg = config(tmp_path, t.raw_dir)
    assert run(cfg) == 1
    files, slices, issues, _ = outputs(cfg)
    assert pairs(issues, "ERROR") == {("TRUNCATED", f"M09_D24_tile{tile}.tif") for tile in ("1-0", "1-1")}
    assert (files["sample_hashes"] == "-;-;-").all()
    assert set(slices.loc[slices["excluded"], "file"]) == {"M09_D24_tile1-0.tif", "M09_D24_tile1-1.tif"}


def test_daylight_saving_timezone(tmp_path):
    """Labels in Europe/London local time across the 2026-10-25 change from BST to GMT."""
    t = synth.make_dataset(tmp_path / "raw", n_slices=40, start=datetime(2026, 10, 25, 0, 0))  # truth in UTC
    london = ZoneInfo("Europe/London")
    for name, zs in t.files.items():
        with tifffile.TiffFile(t.raw_dir / name) as tif:
            data = tif.asarray()
        r, c = name.split("_tile")[1][:3].split("-")
        local = [t.timestamps[z].replace(tzinfo=timezone.utc).astimezone(london) for z in zs]
        synth.write_imagej(t.raw_dir / name, data, [synth.LABEL.format(ts=ts, r=r, c=c) for ts in local])

    cfg = config(tmp_path, t.raw_dir)
    assert run(cfg) == 1
    _, _, issues, _ = outputs(cfg)
    bad = issues[issues["code"] == "NON_MONOTONIC"]
    assert set(bad["file"]) == set(t.files) and bad["message"].str.contains("clocks going back").all()

    cfg = config(tmp_path, t.raw_dir, timezone="Europe/London")
    assert run(cfg) == 0
    _, slices, issues, _ = outputs(cfg)
    assert not pairs(issues, "ERROR") and not pairs(issues, "WARN")
    per_z = slices.drop_duplicates("z").set_index("z")["timestamp"]
    assert per_z.tolist() == [pd.Timestamp(ts) for ts in t.timestamps]
    for name, zs in t.files.items():
        assert slices.loc[slices["file"] == name].sort_values("index")["z"].tolist() == zs


def test_single_slice_files(tmp_path):
    t = synth.make_dataset(tmp_path / "raw", n_slices=2, start=datetime(2026, 9, 24, 23, 58))
    assert all(len(zs) == 1 for zs in t.files.values()) and len(t.files) == 8
    cfg = config(tmp_path, t.raw_dir)
    assert run(cfg) == 0
    _, slices, issues, _ = outputs(cfg)
    assert len(issues) == 0
    assert {(r.file, r.z) for r in slices.itertuples()} == {(n, zs[0]) for n, zs in t.files.items()}


def test_excluded_day_keeps_segment_and_moves_seam(tmp_path):
    """A whole day excluded: no segment split or MISSING_TILE, and its seam moves to the next usable z."""
    t = synth.make_dataset(tmp_path / "raw", n_slices=36, interval_s=7200, start=datetime(2026, 9, 24, 0, 30),
                           gaps={12: 5400}, tile_shape=(64, 80), overlap=(8, 10))
    d25 = day_z(t, "M09_D25")
    assert d25[0] == 12 and "M09_D26_tile0-0.tif" in t.files
    known = [{"file": n, "note": "out of focus", "action": "exclude"} for n in t.files if "_D25_" in n]
    cfg = config(tmp_path, t.raw_dir, known_issues=known)
    assert run(cfg) == 0
    _, slices, issues, _ = outputs(cfg)
    assert issues.loc[issues["severity"] != "INFO", "code"].tolist() == ["TIME_GAP"]
    assert issues.loc[issues["code"] == "TIME_GAP", "z"].tolist() == [12]
    assert (slices["segment"] == 0).all()
    assert slices.loc[slices["excluded"], "z"].unique().tolist() == d25
    usable = load_slices(load_config(cfg))
    assert usable.loc[usable["seam"], "z"].unique().tolist() == [d25[-1] + 1]
