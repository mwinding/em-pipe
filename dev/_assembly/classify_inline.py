"""Sort inline scripts extracted from session transcripts into analysis code worth keeping and the rest.

Kept: scripts that read data or pipeline outputs (TIFF stacks, OME-Zarr, work/ and qc/ tables, NumPy
arrays) and analyse or plot them. Left out, with the reason printed:
  edit     programmatic edits of this repository's own files (pipeline, tests, docs, configs): their
           result is in the git history already
  memory   edits of assistant notes
  movie    movie and promotional-video work (not pipeline development)
  delete   anything that deletes files or cancels jobs
  secret   anything touching credentials or keys
  ops      shell-only job submission and monitoring without analysis code
  trivial  no data access (prints, arithmetic, status checks)

usage: python -I classify_inline.py DIR [--move-kept DST] [--list]
"""
import argparse
import re
import shutil
from pathlib import Path

REPO_FILES = re.compile(r"\b(pipeline|tests|tools|slurm|docs)/[\w./-]+|README\.md|example\.yaml|run_pipeline\.sh|environment\.yml")
WRITES = re.compile(r"write_text\(|\.write\(|sed -i|cat >>?|\breplace\(|open\([^)]*['\"][wa]['\"]")
RULES = [
    ("memory", re.compile(r"/memory/|MEMORY\.md")),
    ("movie", re.compile(r"movie|blender|\bbpy\b|ffmpeg|\.mp4|promo|video|\bmesh(es)?\b|tissue_zoom", re.I)),
    ("delete", re.compile(r"\brm\s+-\w*[rf]|rmtree|\.unlink\(|os\.remove|\bscancel\b|--overwrite")),
    ("secret", re.compile(r"secret|credential|\btoken\b|cloud[v]olume|gcloud|service.account", re.I)),
]
DATA = re.compile(r"tifffile|\bzarr\b|tensorstore|ImageJStack|imagej_tiff|np\.load|pd\.read_csv|read_csv\(|cv2\.|"
                  r"\.ome\.zarr|work/|qc/|thumbs|load_slices|load_config|transforms\.read_csv")


def body_of(text):
    """The script without the extraction header."""
    lines = text.splitlines()
    i = 0
    while i < len(lines) and (lines[i].startswith("# ") or not lines[i].strip()):
        i += 1
    return "\n".join(lines[i:])


def header_field(text, field):
    m = re.search(rf"^# {field}: (.*)$", text, re.M)
    return m.group(1) if m else ""


def classify(text):
    b = body_of(text)
    target, note = header_field(text, "Heredoc line"), header_field(text, "Assistant note before it")
    for name, rx in RULES:
        if rx.search(b) or (name in ("movie", "memory") and rx.search(target + " " + note)):
            return name
    if (REPO_FILES.search(b) and WRITES.search(b)) or re.search(r"(cat|tee) >>?\s*(\S*/)?(pipeline|tests|docs|tools|slurm)/", target):
        return "edit"
    if not re.search(r"^\s*(import |from \w+ import )", b, re.M):
        return "ops"
    if not DATA.search(b):
        return "trivial"
    return "keep"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--move-kept")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    counts = {}
    for p in sorted(Path(a.dir).iterdir()):
        if not p.is_file():
            continue
        c = classify(p.read_text(errors="ignore"))
        counts[c] = counts.get(c, 0) + 1
        if a.list:
            print(f"{c:8s} {p.name}")
        if c == "keep" and a.move_kept:
            Path(a.move_kept).mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, Path(a.move_kept) / p.name)
    print(" ".join(f"{k}={v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
