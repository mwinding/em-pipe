"""The second detector's files (EsB, backscatter) for decurtaining with a reference: which file belongs to
each raw (InLens) file, and whether every one is there with the same slice count and frame size.

Decurtaining with a reference (destreak.reference_from / reference_to) reads, for every raw tile slice, the
same slice of the EsB file of the same day, tile and part. A missing EsB file would otherwise only be noticed
when render starts, hours into a run, so this check runs before anything is submitted (pipeline.preflight,
the ``em-pipe`` command). It reads file names and TIFF headers only.

Naming it recognises (``detect``): an EsB file is named as its raw file with ``_EsB`` (in any capitals,
e.g. ``_ESB``) after the day, and lies in one folder (e.g. ``Backscatter/``) next to the raw file or at the
top of raw_dir:

    M10_D21_tile0-0.tif            Backscatter/M10_D21_EsB_tile0-0.tif
    M10_D21_tile0-0.tif            Backscatter/M10_D21_ESB_tile0-0.tif
    Approach/M09_D18_tile0-0.tif   Approach/Backscatter/M09_D18_EsB_tile0-0.tif
    Approach/M09_D18_tile0-0.tif   Backscatter/M09_D18_ESB_tile0-0.tif          (one flat folder for all)

Raw files in subfolders (e.g. ``Approach/``) are part of the volume like any other: each needs its EsB file.
Joined multi-day stacks (``M11_D12D13_ESB_tile0-0.tif``) are reported with the tool that splits them per day
(tools/split_stack.py).

    python -m pipeline.esb RAW_DIR [--file-pattern REGEX] [--no-headers]     report for a raw folder
"""

import argparse
import os
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from .imagej_tiff import ImageJStack

REFERENCE_FROM = r"^(.*/)?(M\d\d_D\d\d)(.*)$"
_TOKEN = re.compile(r"(?i)_esb(?=[_.])")
_JOINED = re.compile(r"(?i)^M\d\d_D\d\d(?:D\d\d)+_")
_DAY = re.compile(r"^M(\d\d)_D(\d\d)")


def find_candidates(raw_dir):
    """Relative paths of every .tif under raw_dir (all subfolders) whose name has an EsB token."""
    out = []
    for root, dirs, files in os.walk(raw_dir):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for name in sorted(files):
            if name.lower().endswith((".tif", ".tiff")) and _TOKEN.search(name) and not name.startswith("."):
                out.append(os.path.relpath(os.path.join(root, name), raw_dir))
    return out


def _templates(raw, cand):
    """Every reference_to that maps raw file ``raw`` to EsB file ``cand`` (relative paths); none if the pair
    does not follow the naming: the day, then the token, then the rest of the raw name, in one folder next
    to the raw file or at the top of raw_dir (a raw file at the top fits both)."""
    rdir, rname = os.path.split(raw)
    cdir, cname = os.path.split(cand)
    m = _TOKEN.search(cname)
    day = _DAY.match(rname)
    if not m or not day or cname[:m.start()] != day.group(0) or cname[m.end():] != rname[day.end():]:
        return []
    out = []
    for prefix, base in ((r"\1", rdir), ("", "")):
        if cdir == base:
            folder = ""
        elif os.path.dirname(cdir) == base:
            folder = os.path.basename(cdir) + "/"
        else:
            continue
        to = f"{prefix}{folder}\\2{m.group(0)}\\3"
        if re.sub(REFERENCE_FROM, to, raw.replace(os.sep, "/")) == cand.replace(os.sep, "/"):
            out.append(to)
    return out


def detect(raw_files, candidates):
    """(reference_to or None, {another reference_to: number of raw files only it finds an EsB file for}).

    The pattern that finds an EsB file for the most raw files wins (among equals, the one that keeps EsB
    next to each raw folder); patterns that find files the winner does not are returned, so a mixed naming
    can be reported."""
    by_name = defaultdict(list)
    for c in candidates:
        by_name[_TOKEN.sub("", os.path.basename(c), count=1)].append(c)
    found = defaultdict(set)
    for raw in raw_files:
        for c in by_name.get(os.path.basename(raw), []):
            for to in _templates(raw, c):
                found[to].add(raw)
    if not found:
        return None, {}
    best = sorted(found, key=lambda t: (-len(found[t]), not t.startswith(r"\1"), t))[0]
    others = {t: len(files - found[best]) for t, files in found.items() if files - found[best]}
    return best, others


def reference_of(raw, reference_from, reference_to):
    """The EsB file of raw file ``raw`` (relative path), or None when the pattern does not map it."""
    new, n = re.subn(reference_from, reference_to, raw.replace(os.sep, "/"))
    return new if n and new != raw.replace(os.sep, "/") else None


def _folder_days(files):
    """'Approach/M09_D18-M09_D21 (54 files), M09_D22 (9 files)': files grouped by folder and runs of days
    that follow each other in the calendar."""
    groups = defaultdict(list)
    for f in files:
        d, name = os.path.split(f)
        m = _DAY.match(name)
        groups[d].append(((int(m.group(1)), int(m.group(2))) if m else (99, 99), name))
    parts = []
    for d in sorted(groups, key=lambda d: min(groups[d])):
        per_day = Counter(day for day, _ in groups[d])
        days = sorted(per_day)
        run = [days[0]]
        for a, b in zip(days, days[1:] + [None]):
            if b is not None and (b[0] == a[0] and b[1] == a[1] + 1 or b[0] == a[0] + 1 and b[1] == 1 and a[1] >= 28):
                run.append(b)
                continue
            n = sum(per_day[x] for x in run)
            label = lambda x: f"M{x[0]:02d}_D{x[1]:02d}"
            span = label(run[0]) if len(run) == 1 else f"{label(run[0])}-{label(run[-1])}"
            parts.append(f"{d + '/' if d else ''}{span} ({n} file{'s' if n != 1 else ''})")
            run = [b]
    return ", ".join(parts)


@dataclass
class Report:
    """What ``check`` found. ``ok``: every raw file has its EsB file with the same slices and frame size."""
    reference_to: str = None
    n_raw: int = 0
    present: list = field(default_factory=list)       # (raw, esb)
    missing: list = field(default_factory=list)       # raw files without an EsB file
    mismatched: list = field(default_factory=list)    # (raw, esb, what differs)
    unmapped: list = field(default_factory=list)      # raw files the pattern does not map at all
    shared: list = field(default_factory=list)        # (esb, [raw files mapped to it]) when more than one
    joined: list = field(default_factory=list)        # joined multi-day EsB stacks found
    other_naming: dict = field(default_factory=dict)  # {reference_to: files} of EsB files named another way
    headers_read: bool = True

    @property
    def ok(self):
        return bool(self.n_raw) and not (self.missing or self.mismatched or self.unmapped or self.shared)

    def summary(self):
        """One line, e.g. '122 of 185 EsB files present; missing: Approach/M09_D18-M09_D21 (54 files), ...'."""
        s = f"{len(self.present)} of {self.n_raw} EsB files present"
        if self.missing:
            s += f"; {len(self.missing)} missing: {_folder_days(self.missing)}"
        if self.mismatched:
            s += f"; {len(self.mismatched)} not matching their raw file"
        return s

    def lines(self):
        """The full message, one item per line."""
        L = [f"EsB check: {self.summary()}."]
        if self.reference_to:
            L.append(f"  naming: destreak.reference_from '{REFERENCE_FROM}', reference_to '{self.reference_to}'")
        else:
            L.append("  no EsB file was found for any raw file (looked for <day>_EsB<rest of the raw name>, in any "
                     "capitals, in a folder next to the raw files or at the top of raw_dir)")
        for raw, esb, why in self.mismatched[:10]:
            L.append(f"  not matching: {esb}: {why}")
        if len(self.mismatched) > 10:
            L.append(f"  ... {len(self.mismatched) - 10} more not matching")
        if self.unmapped:
            L.append(f"  {len(self.unmapped)} raw files are not mapped by the pattern, e.g. {self.unmapped[0]}")
        for esb, raws in self.shared[:5]:
            L.append(f"  {len(raws)} raw files map to the same EsB file {esb}: {', '.join(raws[:3])}")
        for to, n in self.other_naming.items():
            L.append(f"  {n} EsB files are named another way (reference_to '{to}'): one pattern cannot describe "
                     "both; rename them or move them into one layout")
        if self.joined:
            L.append(f"  joined multi-day EsB stacks found ({len(self.joined)}, e.g. {self.joined[0]}): "
                     "tools/split_stack.py splits them into one stack per day (see its --help)")
        if not self.headers_read:
            L.append("  (file names only: headers not compared)")
        return L


def _shape(path):
    try:
        return ImageJStack(path).shape, None
    except Exception as e:   # anything that stops the header being read
        return None, f"{type(e).__name__}: {e}"


def check(raw_dir, raw_files, reference_to=None, reference_from=REFERENCE_FROM, headers=True, threads=8):
    """Match every raw file to its EsB file. reference_to None: the naming is detected (``detect``).

    With ``headers`` the slice count and frame size of every pair are compared (TIFF headers only)."""
    raw_dir = Path(raw_dir)
    raw_files = sorted(raw_files)
    candidates = find_candidates(raw_dir)
    rep = Report(n_raw=len(raw_files), headers_read=bool(headers))
    rep.joined = [c for c in candidates if _JOINED.match(os.path.basename(c))]
    detected, others = detect(raw_files, candidates)
    if reference_to is None:
        reference_to, rep.other_naming = detected, others
    rep.reference_to = reference_to
    if reference_to is None:
        rep.missing = list(raw_files)
        return rep
    listed = {c.replace(os.sep, "/") for c in candidates}
    targets = defaultdict(list)
    for raw in raw_files:
        esb = reference_of(raw, reference_from, reference_to)
        if esb is None:
            rep.unmapped.append(raw)
        else:
            targets[esb].append(raw)
    pairs = []
    for esb, raws in targets.items():
        if len(raws) > 1:
            rep.shared.append((esb, raws))
        for raw in raws:
            if esb in listed:   # by the folder listing, exact capitals (also on a filesystem that ignores case)
                pairs.append((raw, esb))
            else:
                rep.missing.append(raw)
    rep.missing.sort()
    if headers and pairs:
        paths = [raw_dir / p for pair in pairs for p in pair]
        with ThreadPoolExecutor(max(1, int(threads))) as pool:
            shapes = list(pool.map(_shape, paths))
        for k, (raw, esb) in enumerate(pairs):
            (rs, rerr), (es, eerr) = shapes[2 * k], shapes[2 * k + 1]
            if eerr or rerr:
                rep.mismatched.append((raw, esb, f"header not readable ({eerr or rerr})"))
            elif rs != es:
                rep.mismatched.append((raw, esb, f"{es[0]} slices of {es[1]} x {es[2]} px; its raw file {raw} has "
                                                 f"{rs[0]} slices of {rs[1]} x {rs[2]} px"))
    bad = {raw for raw, _, _ in rep.mismatched}
    rep.present = [(raw, esb) for raw, esb in pairs if raw not in bad]
    return rep


STOP_ADVICE = ("Decurtaining with a reference needs the EsB file of every raw file. Add the missing files and run "
               "again, or state that this run is without decurtaining: destreak.enabled: false in the config "
               "(em-pipe: --no-esb).")


def main(argv=None):
    from .check import scan
    from .config import DEFAULTS
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("raw_dir")
    p.add_argument("--file-pattern", default=DEFAULTS["raw"]["file_pattern"], help="raw.file_pattern of the raw files")
    p.add_argument("--reference-to", help="check this destreak.reference_to instead of detecting the naming")
    p.add_argument("--no-recursive", action="store_true", help="raw files at the top of RAW_DIR only")
    p.add_argument("--no-headers", action="store_true", help="file names only")
    a = p.parse_args(argv)
    if not os.path.isdir(a.raw_dir):
        print(f"raw_dir {a.raw_dir} is not a folder", file=sys.stderr)
        return 1
    found, _ = scan(a.raw_dir, re.compile(a.file_pattern), not a.no_recursive, os.devnull)
    rep = check(a.raw_dir, list(found), a.reference_to, headers=not a.no_headers)
    print("\n".join(rep.lines()))
    if not rep.ok:
        print(STOP_ADVICE)
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main())
