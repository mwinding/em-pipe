"""Split joined raw stacks into one stack per InLens file: joined days, or the parts of one day.

The facility may export a detector as one stack per tile covering several InLens files:

- several acquisition days, e.g. ``M11_D12D13_ESB_tile0-0.tif``, where the InLens raw files are per day
  (``M11_D12_tile0-0.tif``, ``M11_D13_tile0-0.tif``): split into ``M11_D12_EsB_tile0-0.tif`` and
  ``M11_D13_EsB_tile0-0.tif``;
- the parts of one day, e.g. ``M09_D21_ESB_tile0-0.tif``, where InLens has ``M09_D21_tile0-0_part1.tif`` and
  ``M09_D21_tile0-0_part2.tif`` (and no ``M09_D21_tile0-0.tif``): split into ``M09_D21_ESB_tile0-0_part1.tif``
  and ``M09_D21_ESB_tile0-0_part2.tif``, keeping the joined file's detector token, so the splits are named
  like the other EsB files of that export.

A day of a multi-day stack may itself be in parts; each InLens file gets its own split. The joined stack holds
the slices in order, so it splits by the InLens files' slice counts, into the names em-pipe's destreak
reference regex expects (``reference_from: '^(.*/)?(M\\d\\d_D\\d\\d)(.*)$'``,
``reference_to: '\\1Backscatter/\\2_EsB\\3'``; ``python -m pipeline.esb RAW_DIR`` finds the naming). A
one-day stack whose InLens file exists under the same name (an ordinary EsB file), or whose day has no InLens
files in REF_DIR, is not a joined stack and is left alone.

    python tools/split_stack.py plan   JOINED_DIR REF_DIR                 list the splits, check the headers
    python tools/split_stack.py split  JOINED_DIR REF_DIR [--index K] [--group-rw]     write split K (or all)
    python tools/split_stack.py check  JOINED_DIR REF_DIR [--index K]     md5 of split K's pixels and its source's
    python tools/split_stack.py verify JOINED_DIR REF_DIR                 headers, pixels, order, labels; manifest
    python tools/split_stack.py run    JOINED_DIR REF_DIR [--group-rw]    all four, here, in order
    python tools/split_stack.py submit JOINED_DIR REF_DIR [--group-rw] [--parallel N] [--dry-run]
                                                                          the four as a Slurm job chain
Options for every command: --out-dir DIR (default JOINED_DIR), --label NAME (the detector name in the split
names; default EsB for multi-day stacks, the joined file's own token, e.g. ESB, for the parts of one day),
--label-pattern REGEX (the slice labels' pattern, default em-pipe's raw.label_pattern), --manifest PATH
(default OUT_DIR/../<OUT_DIR name>_split.md5; bookkeeping goes to a folder next to it, <manifest>.split_stack/).

Each split is a new ImageJ stack like the raw files (one header, slices stored back to back, the joined stack's
byte order, resolution, unit, spacing and slice labels); its pixel bytes are copied unchanged from the joined
stack. Where the slice labels of the joined stack and of the InLens files both follow the label pattern (e.g.
``..._26-09-21_102225_0-0-0_ESB @ 600 V_raw.tif`` and ``..._26-09-21_102225_0-0-0_InLens_raw.tif``), plan
checks them slice by slice: the same date, time and tile in both, everything but the detector; a difference
stops it before anything is written. verify checks every split's header against its InLens file (slice count,
size, type) and the joined stack, its labels again, that its pixels are byte-identical to the joined stack's
slices (md5, from check), and that the slices are in the right order: at both ends of each split, the split
slice resembles the InLens slice of the same index more than its neighbours (the two detectors record the same
scan). Only then is the manifest (md5 of each split file, md5sum format) written:

    cd OUT_DIR && md5sum -c <manifest>

Safety: it never deletes or changes the joined stacks or the InLens files (read only); a split is written to
<name>.part and renamed when complete, and an existing split is left alone (split skips it).
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import warnings
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.config import DEFAULTS  # noqa: E402
from pipeline.imagej_tiff import ImageJStack  # noqa: E402

CHUNK = 64 << 20
JOINED = re.compile(r"^(?P<m>M\d+)_(?P<days>(?:D\d+)+)_(?P<det>[A-Za-z]+)_(?P<rest>.+\.tiff?)$")
# ImageJ description entries and extra metadata copied from the joined stack (tifffile sets images= itself).
PASS = ("unit", "spacing", "min", "max", "loop", "finterval", "Info", "Ranges", "Properties")
LABEL_PATTERN = DEFAULTS["raw"]["label_pattern"]


def paths(args):
    out = Path(args.out_dir or args.joined_dir).resolve()
    manifest = Path(args.manifest).resolve() if args.manifest else out.parent / f"{out.name}_split.md5"
    return out, manifest, manifest.parent / f"{manifest.name}.split_stack"


def _inlens_files(names, m, d, rest):
    """The InLens files of day ``d`` for a joined stack's ``rest`` (e.g. tile0-0.tif), from the file names of
    REF_DIR: [the day's file], or its parts [..._part1.tif, ..._part2.tif, ...] in order; ([], problem)."""
    if f"{m}_{d}_{rest}" in names:
        return [f"{m}_{d}_{rest}"], None
    stem, ext = os.path.splitext(rest)
    pat = re.compile(re.escape(f"{m}_{d}_{stem}_part") + r"(\d+)" + re.escape(ext))
    parts = sorted((int(mp.group(1)), n) for n in names if (mp := pat.fullmatch(n)))
    if parts and [k for k, _ in parts] != list(range(1, len(parts) + 1)):
        return [], f"its parts are numbered {', '.join(str(k) for k, _ in parts)}, not 1 to {len(parts)}"
    return [n for _, n in parts], None


def _label_keys(labels, label_re):
    """Per slice, what identifies it apart from the detector (date, time, tile, ...); None if any label does
    not follow the pattern."""
    keys = []
    for lb in labels:
        mt = label_re.match(lb)
        if not mt:
            return None
        keys.append(tuple(sorted((k, v) for k, v in mt.groupdict().items() if k != "detector")))
    return keys


def compare_labels(joined_labels, ref_labels, label_re):
    """(compared, [problems]): the joined slices' labels against their InLens slices' labels, slice by slice
    (both lists in the same order). Not compared (False, []) where either has no labels or a label does not
    follow the pattern."""
    if not joined_labels or len(joined_labels) != len(ref_labels):
        return False, []
    a, b = _label_keys(joined_labels, label_re), _label_keys(ref_labels, label_re)
    if a is None or b is None:
        return False, []
    bad = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    if not bad:
        return True, []
    i = bad[0]
    return True, [f"{len(bad)} slice labels differ from the InLens labels (date, time or tile), first at slice {i}: "
                  f"{joined_labels[i]!r} against {ref_labels[i]!r}"]


def plan(joined_dir, ref_dir, out_dir, label=None, quiet=False, label_pattern=LABEL_PATTERN):
    """[{joined, start, n, ref, out}] in a fixed order, after checking every header (and the slice labels where
    they follow ``label_pattern``); exits on a problem. ``label``: the detector name in the split names (None:
    EsB for multi-day stacks, the joined file's own token for the parts of one day)."""
    jd, rd = Path(joined_dir), Path(ref_dir)
    names = {p.name for p in rd.iterdir() if p.is_file()}
    label_re = re.compile(label_pattern)
    items, problems, labels_checked, labels_not = [], [], [], []
    for jp in sorted(p for p in jd.iterdir() if p.is_file() and JOINED.match(p.name)):
        mt = JOINED.match(jp.name)
        days = re.findall(r"D\d+", mt["days"])
        found = {d: _inlens_files(names, mt["m"], d, mt["rest"]) for d in days}
        if len(days) == 1 and len(found[days[0]][0]) < 2 and not found[days[0]][1]:
            continue    # one day, one InLens file of the same name (an ordinary EsB file) or none here: not joined
        odd = [f"{mt['m']}_{d}: {why}" for d, (_, why) in found.items() if why]
        missing = [f"{mt['m']}_{d}_{mt['rest']}" for d, (files, why) in found.items() if not files and not why]
        if odd or missing:
            problems.append(f"{jp.name}: " + "; ".join(odd + ([f"no InLens file {', '.join(missing)} (or its parts) "
                                                                 f"in {rd}"] if missing else [])))
            continue
        refs = [rd / n for d in days for n in found[d][0]]
        js = ImageJStack(jp)
        rs = [ImageJStack(r) for r in refs]
        if not js.contiguous or js.truncated or js.compressed:
            problems.append(f"{jp.name}: not a complete, uncompressed, contiguous stack")
        for r, st in zip(refs, rs):
            if (st.height, st.width, st.dtype.itemsize) != (js.height, js.width, js.dtype.itemsize):
                problems.append(f"{jp.name}: {r.name} is {st.width}x{st.height} {st.dtype}, the joined stack "
                                f"{js.width}x{js.height} {js.dtype}")
        if sum(st.n for st in rs) != js.n:
            problems.append(f"{jp.name}: {js.n} slices, but its InLens files hold "
                            f"{' + '.join(str(st.n) for st in rs)} = {sum(st.n for st in rs)}")
            continue
        ref_labels = [lb for st in rs for lb in (st.labels if len(st.labels) == st.n else [None] * st.n)]
        compared, wrong = compare_labels(js.labels, ref_labels if None not in ref_labels else [], label_re)
        (labels_checked if compared else labels_not).append(jp.name)
        problems += [f"{jp.name}: {w}" for w in wrong]
        name = label or ("EsB" if len(days) > 1 else mt["det"])
        start = 0
        for r, st in zip(refs, rs):
            head = re.match(r"M\d+_D\d+_", r.name).group(0)
            items.append({"joined": str(jp), "start": start, "n": st.n, "ref": str(r),
                          "out": str(Path(out_dir) / f"{head}{name}_{r.name[len(head):]}")})
            start += st.n
    if not items and not problems:
        problems.append(f"no joined stacks (like M11_D12D13_ESB_tile0-0.tif, or M09_D21_ESB_tile0-0.tif where "
                        f"{rd} has M09_D21_tile0-0_part1.tif and _part2.tif) in {jd}")
    if problems:
        sys.exit("split_stack: " + "\n  ".join(["problems, nothing done:"] + problems))
    if not quiet:
        size = sum(it["n"] for it in items) * ImageJStack(items[0]["joined"]).slice_bytes
        print(f"split_stack: {len(items)} splits from {len({it['joined'] for it in items})} joined stacks "
              f"({size / 1e9:.1f} GB of pixels), into {out_dir}")
        for k, it in enumerate(items):
            print(f"  {k}: {Path(it['joined']).name} slices {it['start']}-{it['start'] + it['n'] - 1} "
                  f"-> {Path(it['out']).name} (as {Path(it['ref']).name})")
        if labels_checked:
            print(f"  slice labels: {len(labels_checked)} joined stacks match their InLens files' labels slice by "
                  "slice (date, time, tile)")
        if labels_not:
            print(f"  slice labels: not compared for {len(labels_not)} joined stacks (no labels, or labels not "
                  f"following the label pattern), e.g. {labels_not[0]}")
    return items


def _metadata(js):
    md = {k: js.imagej_metadata[k] for k in PASS if k in js.imagej_metadata}
    return md


def split_one(it, group_rw=False):
    out = Path(it["out"])
    if out.exists():
        print(f"split_stack: {out.name} exists, left alone", flush=True)
        return
    js = ImageJStack(it["joined"])
    part = out.with_name(out.name + ".part")
    md = _metadata(js)
    labels = js.labels
    if len(labels) == js.n:
        md["Labels"] = labels[it["start"]:it["start"] + it["n"]]
    with tifffile.TiffFile(it["joined"]) as tif:
        tags = tif.pages.first.tags
        res = {}
        if "XResolution" in tags and "YResolution" in tags:
            res["resolution"] = (tags["XResolution"].value, tags["YResolution"].value)
            if "ResolutionUnit" in tags:
                res["resolutionunit"] = tags["ResolutionUnit"].value
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*truncating ImageJ file.*")   # one header, as the raw files
        tifffile.imwrite(part, shape=(it["n"], js.height, js.width), dtype=js.dtype, byteorder=js.byteorder,
                         imagej=True, metadata=md, **res)
    ps = ImageJStack(part)
    if not ps.contiguous or ps.n != it["n"] or ps.dtype != js.dtype:
        sys.exit(f"split_stack: {part} came out as {ps.n} slices {ps.dtype}, contiguous={ps.contiguous}")
    total = it["n"] * js.slice_bytes
    with open(it["joined"], "rb", buffering=0) as fi, open(part, "r+b", buffering=0) as fo:
        fi.seek(js.data_offset + it["start"] * js.slice_bytes)
        fo.seek(ps.data_offset)
        buf = bytearray(CHUNK)
        done = 0
        while done < total:
            m = fi.readinto(memoryview(buf)[:min(CHUNK, total - done)])
            if not m:
                sys.exit(f"split_stack: {it['joined']} ended early")
            fo.write(memoryview(buf)[:m])
            done += m
        fo.flush()
        os.fsync(fo.fileno())
    if part.stat().st_size < ps.data_offset + total:   # small stacks keep their per-slice headers after the data
        sys.exit(f"split_stack: {part} is {part.stat().st_size} bytes, less than its {ps.data_offset + total}")
    if group_rw:
        os.chmod(part, 0o660)
    os.replace(part, out)
    print(f"split_stack: wrote {out.name} ({it['n']} slices, {out.stat().st_size} bytes)", flush=True)


def _md5_range(path, offset, length, whole=False):
    """md5 of ``length`` bytes from ``offset``; with whole, also the md5 of the whole file (one pass)."""
    hr, hw = hashlib.md5(), hashlib.md5() if whole else None
    with open(path, "rb") as f:
        if whole:
            head = f.read(offset)
            hw.update(head)
        else:
            f.seek(offset)
        left = length
        while left:
            b = f.read(min(CHUNK, left))
            if not b:
                break
            hr.update(b)
            if whole:
                hw.update(b)
            left -= len(b)
        if whole:
            while b := f.read(CHUNK):
                hw.update(b)
    return hr.hexdigest(), (hw.hexdigest() if whole else None)


def check_one(k, it, work):
    js, os_ = ImageJStack(it["joined"]), ImageJStack(it["out"])
    length = it["n"] * js.slice_bytes
    res = {}
    t = [threading.Thread(target=lambda: res.__setitem__("src", _md5_range(it["joined"], js.data_offset +
                                                                          it["start"] * js.slice_bytes, length))),
         threading.Thread(target=lambda: res.__setitem__("out", _md5_range(it["out"], os_.data_offset, length,
                                                                          whole=True)))]
    for x in t:
        x.start()
    for x in t:
        x.join()
    rec = {"out": it["out"], "pixels_src": res["src"][0], "pixels_out": res["out"][0], "file": res["out"][1]}
    (work / "check").mkdir(parents=True, exist_ok=True)
    tmp = work / "check" / f"{k}.json.tmp"
    tmp.write_text(json.dumps(rec))
    tmp.replace(work / "check" / f"{k}.json")
    print(f"split_stack: pixels of {Path(it['out']).name}: "
          f"{'identical' if rec['pixels_src'] == rec['pixels_out'] else 'DIFFERENT'}", flush=True)


def _corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))


def _detail(st, i, size=1024):
    """The central size x size block of slice i, 2x binned, minus its local mean (fine structure only)."""
    from scipy import ndimage
    s = min(size, st.height, st.width) // 2 * 2
    r0, c0 = (st.height - s) // 2, (st.width - s) // 2
    a = st.read(i, slice(r0, r0 + s), slice(c0, c0 + s)).astype(np.float64)
    a = a.reshape(s // 2, 2, s // 2, 2).mean(axis=(1, 3))
    return a - ndimage.uniform_filter(a, 9)


def day_order(it):
    """[(slice, r same, r previous, r next)] at both ends of a split: InLens slice j against split slices j-1,
    j and j+1 (the joined stack's neighbours where the split has none)."""
    js, rs = ImageJStack(it["joined"]), ImageJStack(it["ref"])
    rows = []
    for j in sorted({0, it["n"] - 1}):
        g = it["start"] + j
        ref = _detail(rs, j)
        r = {d: _corr(ref, _detail(js, g + d)) if 0 <= g + d < js.n else None for d in (-1, 0, 1)}
        rows.append((j, r[0], r[-1], r[1]))
    return rows


def verify(items, work, manifest, out_dir, label_pattern=LABEL_PATTERN):
    problems, lines, report = [], [], []
    label_re, n_labels = re.compile(label_pattern), 0
    for k, it in enumerate(items):
        name = Path(it["out"]).name
        if not Path(it["out"]).exists():
            problems.append(f"{name}: not written")
            continue
        js, rs, os_ = ImageJStack(it["joined"]), ImageJStack(it["ref"]), ImageJStack(it["out"])
        if (os_.n, os_.height, os_.width) != (rs.n, rs.height, rs.width):
            problems.append(f"{name}: {os_.shape}, its InLens file {rs.shape}")
        if os_.dtype != js.dtype or not os_.contiguous or os_.truncated:
            problems.append(f"{name}: {os_.dtype} contiguous={os_.contiguous} truncated={os_.truncated}")
        if os_.voxel_size_nm != js.voxel_size_nm:
            problems.append(f"{name}: voxel size {os_.voxel_size_nm}, the joined stack {js.voxel_size_nm}")
        if len(js.labels) == js.n and os_.labels != js.labels[it["start"]:it["start"] + it["n"]]:
            problems.append(f"{name}: slice labels differ from the joined stack's")
        compared, wrong = compare_labels(os_.labels, rs.labels, label_re)
        n_labels += compared
        problems += [f"{name}: {w}" for w in wrong]
        cf = work / "check" / f"{k}.json"
        if not cf.exists():
            problems.append(f"{name}: no pixel check (run check)")
            continue
        rec = json.loads(cf.read_text())
        if rec["pixels_src"] != rec["pixels_out"]:
            problems.append(f"{name}: pixels differ from the joined stack's slices")
        for j, same, prev, nxt in day_order(it):
            others = [v for v in (prev, nxt) if v is not None]
            ok = all(same > v for v in others)
            report.append(f"  {name} slice {j}: r = {same:.3f} with InLens slice {j}; neighbours "
                          + ", ".join(f"{v:.3f}" for v in others) + ("" if ok else "  <- NOT the best match"))
            if not ok:
                problems.append(f"{name}: slice {j} matches a neighbouring InLens slice better (days or parts out "
                                "of order?)")
        lines.append(f"{rec['file']}  ./{Path(it['out']).relative_to(out_dir).as_posix()}")
    print("split_stack: slice order (fine-structure correlation, split slice vs InLens):", *report, sep="\n")
    print(f"split_stack: slice labels (date, time, tile) compared with the InLens file's, slice by slice: "
          f"{n_labels} of {len(items)} splits" + ("" if n_labels == len(items) else
                                                   " (the others have no labels following the label pattern)"))
    if problems:
        print("split_stack: FAIL", *problems, "no manifest written", sep="\n  ")
        return 1
    manifest.write_text("\n".join(lines) + "\n")
    print(f"split_stack: PASS: {len(items)} splits match their InLens files and the joined stacks' pixels. "
          f"Manifest: {manifest}\n  check again later: cd {out_dir} && md5sum -c {manifest}")
    return 0


def indices(args, n):
    if args.index is not None:
        return [args.index]
    if "SLURM_ARRAY_TASK_ID" in os.environ:
        return [int(os.environ["SLURM_ARRAY_TASK_ID"])]
    return list(range(n))


def submit(args, n):
    here = Path(__file__).resolve().parents[1]
    common = [str(Path(args.joined_dir).resolve()), str(Path(args.ref_dir).resolve())]
    for opt in ("out_dir", "label", "label_pattern", "manifest"):
        v = getattr(args, opt)
        if v:
            common += [f"--{opt.replace('_', '-')}", str(Path(v).resolve()) if opt in ("out_dir", "manifest") else v]
    sb = ["sbatch", "--parsable", "--chdir", str(here)]
    arr = f"--array=0-{n - 1}%{args.parallel}"
    spl = sb + ["--job-name=split-stack-split", arr, "slurm/split_stack.sbatch", "split", *common] + \
        (["--group-rw"] if args.group_rw else [])
    chk = sb + ["--job-name=split-stack-check", arr, "slurm/split_stack.sbatch", "check", *common]
    ver = sb + ["--job-name=split-stack-verify", "slurm/split_stack.sbatch", "verify", *common]
    if args.dry_run:
        for c in (spl, chk, ver):
            print(" ".join(c))
        return 0
    j1 = subprocess.run(spl, check=True, capture_output=True, text=True).stdout.strip().split(";")[0]
    j2 = subprocess.run(chk[:2] + [f"--dependency=afterok:{j1}"] + chk[2:], check=True, capture_output=True,
                        text=True).stdout.strip().split(";")[0]
    j3 = subprocess.run(ver[:2] + [f"--dependency=afterany:{j2}"] + ver[2:], check=True, capture_output=True,
                        text=True).stdout.strip().split(";")[0]
    print(f"split_stack: split {j1}, check {j2}, verify {j3} (its log says PASS or FAIL)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("plan", "split", "check", "verify", "run", "submit"))
    ap.add_argument("joined_dir", help="folder of the joined stacks (e.g. M11_D12D13_ESB_tile0-0.tif)")
    ap.add_argument("ref_dir", help="folder of the InLens stacks (e.g. M11_D12_tile0-0.tif, M09_D21_tile0-0_part1.tif)")
    ap.add_argument("--out-dir", help="where the splits go (default: JOINED_DIR)")
    ap.add_argument("--label", help="detector name in the split names (default: EsB for multi-day stacks, the "
                                    "joined file's own token for the parts of one day)")
    ap.add_argument("--label-pattern", help="regex of the slice labels (default: em-pipe's raw.label_pattern); "
                                            "every named group but detector must agree with InLens")
    ap.add_argument("--manifest", help="manifest path (default OUT_DIR/../<OUT_DIR name>_split.md5)")
    ap.add_argument("--index", type=int, help="one split (default: the Slurm array task, else all)")
    ap.add_argument("--group-rw", action="store_true", help="splits rw for owner and group only (0660)")
    ap.add_argument("--parallel", type=int, default=6, help="submit: array tasks running at once (default 6)")
    ap.add_argument("--dry-run", action="store_true", help="submit: print the sbatch commands only")
    a = ap.parse_args(argv)
    out, manifest, work = paths(a)
    for d in (Path(a.joined_dir).resolve(), Path(a.ref_dir).resolve()):
        if not d.is_dir():
            sys.exit(f"split_stack: {d} is not a folder")
    lp = a.label_pattern or LABEL_PATTERN
    items = plan(a.joined_dir, a.ref_dir, out, a.label, quiet=a.cmd not in ("plan", "submit", "run"),
                 label_pattern=lp)
    if a.cmd == "plan":
        return 0
    if a.cmd == "submit":
        return submit(a, len(items))
    out.mkdir(parents=True, exist_ok=True)
    if a.cmd in ("split", "run"):
        for k in (range(len(items)) if a.cmd == "run" else indices(a, len(items))):
            split_one(items[k], a.group_rw)
    if a.cmd in ("check", "run"):
        for k in (range(len(items)) if a.cmd == "run" else indices(a, len(items))):
            check_one(k, items[k], work)
    if a.cmd in ("verify", "run"):
        return verify(items, work, manifest, out, lp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
