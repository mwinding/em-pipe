"""Read-only: duplicate/label check of 37i InLens (re-export) and EsB files, run from the em-pipe clone."""
import json, os, re, sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, ".")
from pipeline.check import read_header

RAW = Path(os.path.expandvars("${DEV_RAW_37I}"))
CACHE = Path(os.path.expandvars("${DEV_37I}/work/check/cache"))
NAME = re.compile(r"^M(?P<month>\d{2})_D(?P<day>\d{2})(?P<esb>_EsB)?(?:_tile(?P<row>\d+)-(?P<col>\d+))?"
                  r"(?:_part(?P<part>\d+))?\.tif$")
LABEL = re.compile(r"^(?P<instrument>.+?)_(?P<date>\d{2}-\d{2}-\d{2})_(?P<time>\d{6})_(?P<group>\d+)-(?P<row>\d+)"
                   r"-(?P<col>\d+)_(?P<detector>[^_]+)(?:_raw)?\.tif$")
HB = 1048576

files = {}
for root, dirs, names in os.walk(RAW):
    dirs[:] = [d for d in dirs if d != "uCT datasets"]
    for n in names:
        m = NAME.match(n)
        if m:
            rel = os.path.relpath(os.path.join(root, n), RAW)
            files[rel] = m.groupdict()
inlens = sorted(r for r, g in files.items() if not g["esb"])
esb = sorted(r for r, g in files.items() if g["esb"])
print(f"files: {len(inlens)} InLens, {len(esb)} EsB")


def cached(rel):
    st = (RAW / rel).stat()
    p = CACHE / (rel.replace(os.sep, "__") + ".json")
    if p.exists():
        e = json.loads(p.read_text())
        k = e["key"]
        if k["size"] == st.st_size and k["mtime_ns"] == st.st_mtime_ns and k["hash_bytes"] == HB:
            return e["header"], "cache"
    return read_header(RAW / rel, HB), "read"


with ThreadPoolExecutor(8) as pool:
    res = dict(zip(inlens + esb, pool.map(cached, inlens + esb)))
hdr = {r: h for r, (h, _) in res.items()}
fresh = [r for r, (_, how) in res.items() if how == "read"]
print(f"read fresh: {len(fresh)} ({len([r for r in fresh if r in inlens])} InLens: "
      f"{[r for r in fresh if r in inlens]})")


def parsed(rel):
    out = []
    for lab in hdr[rel]["labels"]:
        m = LABEL.match(lab)
        out.append(None if m is None else (m["date"], m["time"], f"{int(m['row'])}-{int(m['col'])}", m["detector"]))
    return out


def name_tile(rel):
    g = files[rel]
    return None if g["row"] is None else f"{int(g['row'])}-{int(g['col'])}"


print("\n== re-exported InLens M11_D04_tile0-0_part1.tif ==")
r0, r1 = "M11_D04_tile0-0_part1.tif", "M11_D04_tile0-1_part1.tif"
p0, p1 = parsed(r0), parsed(r1)
print("slices", hdr[r0]["n_slices"], "complete", hdr[r0]["n_complete"], "| tile0-1:", hdr[r1]["n_slices"])
print("label tiles", Counter(p[2] if p else None for p in p0), "detectors", Counter(p[3] if p else None for p in p0))
print("label times equal to tile 0-1's:", [p[:2] for p in p0 if p] == [p[:2] for p in p1 if p])
print("hashes 0-0", hdr[r0]["sample_hashes"])
print("hashes 0-1", hdr[r1]["sample_hashes"])

print("\n== re-exported InLens M11_D10 tiles 0-1, 1-0, 1-1 (vs tile 0-0 and their 12 Nov cache) ==")
ref = parsed("M11_D10_tile0-0.tif")
for t in ("0-1", "1-0", "1-1"):
    r = f"M11_D10_tile{t}.tif"
    p = parsed(r)
    old = json.loads((CACHE / (r + ".json")).read_text())["header"]
    print(f"{r}: {hdr[r]['n_slices']} slices (was {old['n_slices']}), unparsed {sum(x is None for x in p)}, "
          f"tiles {dict(Counter(x[2] for x in p if x))}, detectors {dict(Counter(x[3] for x in p if x))}, "
          f"[583] {p[583] if len(p) > 583 else None}, times = tile 0-0's: "
          f"{[x[:2] if x else None for x in p] == [x[:2] if x else None for x in ref]}, "
          f"pixel samples unchanged: {hdr[r]['sample_hashes'] == old['sample_hashes']}")

print("\n== InLens files: unparsed labels, tile mismatches, detectors ==")
bad_inl = 0
for r in inlens:
    p = parsed(r)
    nt = name_tile(r)
    tiles = Counter(x[2] for x in p if x)
    dets = Counter(x[3] for x in p if x)
    unp = sum(x is None for x in p)
    if unp or (nt and set(tiles) - {nt}) or set(dets) != {"InLens"} or hdr[r]["n_complete"] < hdr[r]["n_slices"]:
        bad_inl += 1
        print(r, unp, dict(tiles), dict(dets), hdr[r]["n_complete"], hdr[r]["n_slices"])
print(f"{len(inlens) - bad_inl} of {len(inlens)} InLens files OK")

print("\n== re-exported InLens M11_D10 tiles 0-1, 1-0, 1-1 (vs tile 0-0 and their 12 Nov cache) ==")
ref = parsed("M11_D10_tile0-0.tif")
for t in ("0-1", "1-0", "1-1"):
    r = f"M11_D10_tile{t}.tif"
    p = parsed(r)
    old = json.loads((CACHE / (r + ".json")).read_text())["header"]
    print(f"{r}: {hdr[r]['n_slices']} slices (was {old['n_slices']}), unparsed {sum(x is None for x in p)}, "
          f"tiles {dict(Counter(x[2] for x in p if x))}, detectors {dict(Counter(x[3] for x in p if x))}, "
          f"[583] {p[583] if len(p) > 583 else None}, times = tile 0-0's: "
          f"{[x[:2] if x else None for x in p] == [x[:2] if x else None for x in ref]}, "
          f"pixel samples unchanged: {hdr[r]['sample_hashes'] == old['sample_hashes']}")

print("\n== InLens files: unparsed labels, tile mismatches, detectors ==")
bad_inl = 0
for r in inlens:
    p = parsed(r)
    nt = name_tile(r)
    tiles = Counter(x[2] for x in p if x)
    dets = Counter(x[3] for x in p if x)
    unp = sum(x is None for x in p)
    if unp or (nt and set(tiles) - {nt}) or set(dets) != {"InLens"} or hdr[r]["n_complete"] < hdr[r]["n_slices"]:
        bad_inl += 1
        print(r, unp, dict(tiles), dict(dets), hdr[r]["n_complete"], hdr[r]["n_slices"])
print(f"{len(inlens) - bad_inl} of {len(inlens)} InLens files OK")

print("\n== identical sampled slices (any md5 shared by >1 file, InLens + EsB) ==")
by = defaultdict(list)
for r, h in hdr.items():
    for i, x in enumerate(h["sample_hashes"].split(";")):
        if x != "-":
            by[x].append(f"{r}[{('first', 'mid', 'last')[i]}]")
shared = {k: v for k, v in by.items() if len(v) > 1}
print("none" if not shared else "\n".join(", ".join(v) for v in shared.values()))
blank = {r: h["sample_hashes"] for r, h in hdr.items() if "-" in h["sample_hashes"].split(";")}
print("files with a blank/unusable sample:", blank or "none")

print("\n== EsB files: labels and match to their InLens file ==")
problems = 0
for r in esb:
    h, p = hdr[r], parsed(r)
    tiles = Counter(x[2] for x in p if x)
    dets = Counter(x[3] for x in p if x)
    bad = sum(x is None for x in p)
    nt = name_tile(r)
    twin = os.path.join(os.path.dirname(os.path.dirname(r)), os.path.basename(r).replace("_EsB", ""))
    twin = os.path.normpath(twin)
    notes = []
    if bad:
        notes.append(f"{bad} unparsed labels (e.g. {[l for l, x in zip(h['labels'], p) if x is None][:2]})")
    if nt is not None and set(tiles) - {nt}:
        notes.append(f"labels say tiles {dict(tiles)}, name says {nt}")
    if nt is None and len(tiles) != 1:
        notes.append(f"single-tile file with label tiles {dict(tiles)}")
    if not dets or not all(d.upper().startswith("ESB") for d in dets):
        notes.append(f"detectors {dict(dets)}")
    if h["n_complete"] < h["n_slices"]:
        notes.append(f"truncated: {h['n_complete']} of {h['n_slices']} complete")
    if twin not in hdr:
        notes.append(f"no InLens file {twin}")
    else:
        q = parsed(twin)
        if h["n_slices"] != hdr[twin]["n_slices"]:
            notes.append(f"{h['n_slices']} slices vs InLens {hdr[twin]['n_slices']}")
        if (h["height"], h["width"]) != (hdr[twin]["height"], hdr[twin]["width"]):
            notes.append(f"size {h['height']}x{h['width']} vs InLens {hdr[twin]['height']}x{hdr[twin]['width']}")
        a = [x[:3] if x else None for x in p]
        b = [x[:3] if x else None for x in q]
        if a != b:
            diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
            notes.append(f"label time/tile differs from InLens at {len(diff)} slices (first {diff[:3]}: "
                         f"{[a[i] for i in diff[:2]]} vs {[b[i] for i in diff[:2]]})")
    if notes:
        problems += 1
        print(r, "|", tiles and dict(tiles), "|", "; ".join(notes))
twins = {os.path.normpath(os.path.join(os.path.dirname(os.path.dirname(r)), os.path.basename(r).replace("_EsB", "")))
         for r in esb}
print(f"{len(esb) - problems} of {len(esb)} EsB files OK; InLens files without an EsB file:",
      sorted(set(inlens) - twins) or "none")
print("EsB slices total", sum(hdr[r]["n_slices"] for r in esb), "InLens slices total",
      sum(hdr[r]["n_slices"] for r in inlens))
