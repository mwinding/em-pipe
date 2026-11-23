"""Split joined multi-day raw stacks into one stack per day that matches the per-day InLens files.

The facility may export a detector as one stack per tile covering several acquisition days, e.g.
``M11_D12D13_ESB_tile0-0.tif``, while the InLens raw files are per day: ``M11_D12_tile0-0.tif`` and
``M11_D13_tile0-0.tif``. The joined stack holds the days' slices in order, so it splits by the InLens files'
slice counts into ``M11_D12_EsB_tile0-0.tif`` and ``M11_D13_EsB_tile0-0.tif``: the names em-pipe's
destreak reference regex expects (``reference_from: '^(.*/)?(M\\d\\d_D\\d\\d)(.*)$'``,
``reference_to: '\\1Backscatter/\\2_EsB\\3'``).

    python tools/split_stack.py plan   JOINED_DIR REF_DIR                 list the splits, check the headers
    python tools/split_stack.py split  JOINED_DIR REF_DIR [--index K] [--group-rw]     write split K (or all)
    python tools/split_stack.py check  JOINED_DIR REF_DIR [--index K]     md5 of split K's pixels and its source's
    python tools/split_stack.py verify JOINED_DIR REF_DIR                 headers, pixels, day order; manifest
    python tools/split_stack.py run    JOINED_DIR REF_DIR [--group-rw]    all four, here, in order
    python tools/split_stack.py submit JOINED_DIR REF_DIR [--group-rw] [--parallel N] [--dry-run]
                                                                          the four as a Slurm job chain
Options for every command: --out-dir DIR (default JOINED_DIR), --label NAME (default EsB), --manifest PATH
(default OUT_DIR/../<OUT_DIR name>_split.md5; bookkeeping goes to a folder next to it, <manifest>.split_stack/).

Each split is a new ImageJ stack like the raw files (one header, slices stored back to back, the joined stack's
byte order, resolution, unit, spacing and slice labels); its pixel bytes are copied unchanged from the joined
stack. verify checks every split's header against its InLens file (slice count, size, type) and the joined
stack, that its pixels are byte-identical to the joined stack's slices (md5, from check), and that the days are
in the right order: at both ends of each day, the split slice resembles the InLens slice of the same index more
than its neighbours (the two detectors record the same scan). Only then is the manifest (md5 of each split
file, md5sum format) written:

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
from pipeline.imagej_tiff import ImageJStack  # noqa: E402

CHUNK = 64 << 20
JOINED = re.compile(r"^(?P<m>M\d+)_(?P<days>(?:D\d+){2,})_(?P<det>[A-Za-z]+)_(?P<rest>.+\.tiff?)$")
# ImageJ description entries and extra metadata copied from the joined stack (tifffile sets images= itself).
PASS = ("unit", "spacing", "min", "max", "loop", "finterval", "Info", "Ranges", "Properties")


def paths(args):
    out = Path(args.out_dir or args.joined_dir).resolve()
    manifest = Path(args.manifest).resolve() if args.manifest else out.parent / f"{out.name}_split.md5"
    return out, manifest, manifest.parent / f"{manifest.name}.split_stack"


def plan(joined_dir, ref_dir, out_dir, label="EsB", quiet=False):
    """[{joined, start, n, ref, out}] in a fixed order, after checking every header; exits on a problem."""
    jd, rd = Path(joined_dir), Path(ref_dir)
    items, problems = [], []
    for jp in sorted(p for p in jd.iterdir() if p.is_file() and JOINED.match(p.name)):
        mt = JOINED.match(jp.name)
        days = re.findall(r"D\d+", mt["days"])
        refs = [rd / f"{mt['m']}_{d}_{mt['rest']}" for d in days]
        missing = [r.name for r in refs if not r.exists()]
        if missing:
            problems.append(f"{jp.name}: no InLens file {', '.join(missing)} in {rd}")
            continue
        js = ImageJStack(jp)
        rs = [ImageJStack(r) for r in refs]
        if not js.contiguous or js.truncated or js.compressed:
            problems.append(f"{jp.name}: not a complete, uncompressed, contiguous stack")
        for r, st in zip(refs, rs):
            if (st.height, st.width, st.dtype.itemsize) != (js.height, js.width, js.dtype.itemsize):
                problems.append(f"{jp.name}: {r.name} is {st.width}x{st.height} {st.dtype}, the joined stack "
                                f"{js.width}x{js.height} {js.dtype}")
        if sum(st.n for st in rs) != js.n:
            problems.append(f"{jp.name}: {js.n} slices, but the InLens days hold "
                            f"{' + '.join(str(st.n) for st in rs)} = {sum(st.n for st in rs)}")
            continue
        start = 0
        for d, r, st in zip(days, refs, rs):
            items.append({"joined": str(jp), "start": start, "n": st.n, "ref": str(r),
                          "out": str(Path(out_dir) / f"{mt['m']}_{d}_{label}_{mt['rest']}")})
            start += st.n
    if not items and not problems:
        problems.append(f"no joined stacks (like M11_D12D13_ESB_tile0-0.tif) in {jd}")
    if problems:
        sys.exit("split_stack: " + "\n  ".join(["problems, nothing done:"] + problems))
    if not quiet:
        size = sum(it["n"] for it in items) * ImageJStack(items[0]["joined"]).slice_bytes
        print(f"split_stack: {len(items)} splits from {len({it['joined'] for it in items})} joined stacks "
              f"({size / 1e9:.1f} GB of pixels), into {out_dir}")
        for k, it in enumerate(items):
            print(f"  {k}: {Path(it['joined']).name} slices {it['start']}-{it['start'] + it['n'] - 1} "
                  f"-> {Path(it['out']).name} (as {Path(it['ref']).name})")
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


def verify(items, work, manifest, out_dir):
    problems, lines, report = [], [], []
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
                problems.append(f"{name}: slice {j} matches a neighbouring InLens slice better (days out of order?)")
        lines.append(f"{rec['file']}  ./{Path(it['out']).relative_to(out_dir).as_posix()}")
    print("split_stack: day order (fine-structure correlation, split slice vs InLens):", *report, sep="\n")
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
    for opt in ("out_dir", "label", "manifest"):
        v = getattr(args, opt)
        if v:
            common += [f"--{opt.replace('_', '-')}", str(Path(v).resolve()) if opt != "label" else v]
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
    ap.add_argument("ref_dir", help="folder of the per-day InLens stacks (e.g. M11_D12_tile0-0.tif)")
    ap.add_argument("--out-dir", help="where the splits go (default: JOINED_DIR)")
    ap.add_argument("--label", default="EsB", help="detector name in the split names (default EsB)")
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
    items = plan(a.joined_dir, a.ref_dir, out, a.label, quiet=a.cmd not in ("plan", "submit", "run"))
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
        return verify(items, work, manifest, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
