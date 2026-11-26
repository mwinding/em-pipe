"""Copy raw data files to another place and prove the copies are byte-identical (md5), then leave a
checksum manifest next to the copy.

    python tools/copy_verify.py plan   SRC DST [--pattern REGEX]       list the files to copy
    python tools/copy_verify.py copy   SRC DST [--index K] [--group-rw] copy file K (or every file)
    python tools/copy_verify.py md5    SRC DST [--index K]             md5 of file K on both sides
    python tools/copy_verify.py verify SRC DST [--manifest PATH]       compare; write the manifest if all match
    python tools/copy_verify.py run    SRC DST [--pattern REGEX] [--group-rw]   all four, here, in order
    python tools/copy_verify.py submit SRC DST [--pattern REGEX] [--group-rw] [--parallel N] [--dry-run]
                                                                       the four as a Slurm job chain

SRC and DST are folders; files are matched by REGEX against their path relative to SRC (default: every
.tif), searched recursively, and keep that relative path under DST. The manifest, written only when every
file matches, is DST.md5 next to DST (or --manifest PATH, e.g. next to the raw folder when DST is a folder
inside it), in md5sum format, so a copy can be checked again later with

    cd DST && md5sum -c <manifest>

It covers every file in DST (a file in DST that is not in the list fails verify), so a later run over the
whole folder replaces it: one list per copied folder. While the chain runs, its bookkeeping (the file list
and per-file checksums) is in DST.copy_verify/ next to DST; after a PASS that folder is removed (its
checksums are in the manifest), after a FAIL it is kept to show what went wrong.

Safety: it never deletes data (rsync without --delete; a failed copy leaves rsync's partial file for the
next try; only its own bookkeeping folder is removed); it refuses SRC == DST, DST inside SRC and SRC inside
DST; source files are only read. --group-rw makes the copies rw for owner and group, nothing for others
(0660), like a lab share.

The record of which data were copied where (paths, dates, the manifest's location) belongs in the
lab's own configuration repository, not here.
"""
import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path

CHUNK = 64 << 20


def check_paths(src, dst):
    s, d = Path(src).resolve(), Path(dst).resolve()
    if not s.is_dir():
        sys.exit(f"copy_verify: SRC {src} is not a folder")
    if s == d:
        sys.exit("copy_verify: SRC and DST are the same folder")
    if d == s or s in d.parents:
        sys.exit(f"copy_verify: DST {dst} is inside SRC {src}")
    if d in s.parents:
        sys.exit(f"copy_verify: SRC {src} is inside DST {dst}")
    return s, d


def work_dir(dst):
    d = Path(dst).resolve()
    return d.parent / f"{d.name}.copy_verify"


def manifest_path(dst, manifest=None):
    if manifest:
        return Path(manifest).resolve()
    d = Path(dst).resolve()
    return d.parent / f"{d.name}.md5"


def plan(src, dst, pattern=r".*\.tif$"):
    s, d = check_paths(src, dst)
    rx = re.compile(pattern)
    files = sorted(p.relative_to(s).as_posix() for p in s.rglob("*") if p.is_file() and rx.search(p.relative_to(s).as_posix()))
    if not files:
        sys.exit(f"copy_verify: no files under {src} match {pattern!r}")
    w = work_dir(d)
    w.mkdir(parents=True, exist_ok=True)
    (w / "files.txt").write_text("\n".join(files) + "\n")
    print(f"copy_verify: {len(files)} files to copy ({sum((s / f).stat().st_size for f in files) / 1e9:.1f} GB); list in {w / 'files.txt'}")
    return files


def file_list(dst):
    p = work_dir(dst) / "files.txt"
    if not p.exists():
        sys.exit(f"copy_verify: {p} not found: run plan first")
    return [f for f in p.read_text().splitlines() if f]


def indices(args, n):
    if args.index is not None:
        return [args.index]
    if "SLURM_ARRAY_TASK_ID" in os.environ:
        return [int(os.environ["SLURM_ARRAY_TASK_ID"])]
    return list(range(n))


def copy_one(src, dst, rel, group_rw=False):
    a, b = Path(src) / rel, Path(dst) / rel
    b.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("rsync"):
        subprocess.run(["rsync", "-t", "--partial", str(a), str(b)], check=True)
    else:
        shutil.copy2(a, b)
    if a.stat().st_size != b.stat().st_size:
        sys.exit(f"copy_verify: size mismatch after copying {rel}: {a.stat().st_size} vs {b.stat().st_size}")
    if group_rw:
        os.chmod(b, 0o660)
    print(f"copy_verify: copied {rel} ({b.stat().st_size} bytes)", flush=True)


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def md5_one(src, dst, k, rel):
    out = work_dir(dst) / "md5"
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    threads = [threading.Thread(target=lambda side, root: res.__setitem__(side, md5_file(Path(root) / rel)), args=(side, root))
               for side, root in (("src", src), ("dst", dst))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for side in ("src", "dst"):
        tmp = out / f"{side}_{k}.md5.tmp"
        tmp.write_text(f"{res[side]}  ./{rel}\n")
        tmp.replace(out / f"{side}_{k}.md5")
    print(f"copy_verify: md5 {rel}: {'match' if res['src'] == res['dst'] else 'MISMATCH'}", flush=True)


def verify(src, dst, manifest=None):
    s, d = check_paths(src, dst)
    files, out = file_list(d), work_dir(d) / "md5"
    problems, lines = [], []
    for k, rel in enumerate(files):
        got = {}
        for side in ("src", "dst"):
            p = out / f"{side}_{k}.md5"
            got[side] = p.read_text().split()[0] if p.exists() and p.read_text().strip() else None
        if None in got.values():
            problems.append(f"no checksum for {rel} ({'source' if got['src'] is None else 'copy'})")
        elif got["src"] != got["dst"]:
            problems.append(f"checksum mismatch: {rel}")
        else:
            lines.append(f"{got['dst']}  ./{rel}")
    listed = set(files)
    extra = sorted(p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file() and p.relative_to(d).as_posix() not in listed)
    problems += [f"extra file in DST, not in the list: {e}" for e in extra]
    if problems:
        print("copy_verify: FAIL", *problems, "no manifest written", sep="\n  ")
        return 1
    m = manifest_path(d, manifest)
    tmp = m.with_name(m.name + ".part")
    tmp.write_text("\n".join(lines) + "\n")
    tmp.replace(m)
    shutil.rmtree(work_dir(d))            # its checksums are in the manifest now
    print(f"copy_verify: PASS: all {len(files)} files byte-identical (md5). Manifest: {m}\n"
          f"  check again later: cd {d} && md5sum -c {m}")
    return 0


def submit(args):
    files = plan(args.src, args.dst, args.pattern)
    n, here = len(files), Path(__file__).resolve().parents[1]
    sb = ["sbatch", "--parsable", "--chdir", str(here)]
    common = [str(Path(args.src).resolve()), str(Path(args.dst).resolve())]
    vopt = ["--manifest", str(Path(args.manifest).resolve())] if args.manifest else []
    cpy = sb + ["--job-name=copy-verify-copy", f"--array=0-{n - 1}%{args.parallel}", "slurm/copy_verify.sbatch", "copy",
                *common] + (["--group-rw"] if args.group_rw else [])
    md5 = sb + ["--job-name=copy-verify-md5", f"--array=0-{n - 1}%{args.parallel}", "slurm/copy_verify.sbatch", "md5", *common]
    ver = sb + ["--job-name=copy-verify-check", "slurm/copy_verify.sbatch", "verify", *common, *vopt]
    if args.dry_run:
        for c in (cpy, md5, ver):
            print(" ".join(c))
        return 0
    j1 = subprocess.run(cpy, check=True, capture_output=True, text=True).stdout.strip().split(";")[0]
    j2 = subprocess.run(md5[:2] + [f"--dependency=afterok:{j1}"] + md5[2:], check=True, capture_output=True, text=True).stdout.strip().split(";")[0]
    j3 = subprocess.run(ver[:2] + [f"--dependency=afterany:{j2}"] + ver[2:], check=True, capture_output=True, text=True).stdout.strip().split(";")[0]
    print(f"copy_verify: copy {j1}, md5 {j2}, verify {j3} (its log says PASS or FAIL)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("plan", "copy", "md5", "verify", "run", "submit"))
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--pattern", default=r".*\.tif$", help="regex on paths relative to SRC (default: .tif files)")
    ap.add_argument("--index", type=int, help="one file of the list (default: the Slurm array task, else all)")
    ap.add_argument("--group-rw", action="store_true", help="copies rw for owner and group only (0660)")
    ap.add_argument("--parallel", type=int, default=6, help="submit: array tasks running at once (default 6)")
    ap.add_argument("--dry-run", action="store_true", help="submit: print the sbatch commands only")
    ap.add_argument("--manifest", help="where the checksum list goes (default DST.md5 next to DST)")
    a = ap.parse_args(argv)
    check_paths(a.src, a.dst)
    if a.cmd == "plan":
        plan(a.src, a.dst, a.pattern)
        return 0
    if a.cmd == "submit":
        return submit(a)
    if a.cmd == "run":
        files = plan(a.src, a.dst, a.pattern)
        for rel in files:
            copy_one(a.src, a.dst, rel, a.group_rw)
        for k, rel in enumerate(files):
            md5_one(a.src, a.dst, k, rel)
        return verify(a.src, a.dst, a.manifest)
    files = file_list(a.dst)
    if a.cmd == "verify":
        return verify(a.src, a.dst, a.manifest)
    for k in indices(a, len(files)):
        if a.cmd == "copy":
            copy_one(a.src, a.dst, files[k], a.group_rw)
        else:
            md5_one(a.src, a.dst, k, files[k])
    return 0


if __name__ == "__main__":
    sys.exit(main())
