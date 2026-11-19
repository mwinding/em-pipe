"""Read-only, 37i: the extra EsB slices (D24, D02), slice 583 of D10, and a denser copy check (8 slices/file)."""
import hashlib, os, re, sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from pipeline.imagej_tiff import ImageJStack

RAW = Path(os.path.expandvars("${DEV_RAW_37I}"))
NAME = re.compile(r"^M\d{2}_D\d{2}(_EsB)?(?:_tile\d+-\d+)?(?:_part\d+)?\.tif$")
TILES = ["0-0", "0-1", "1-0", "1-1"]


def stats(st, i):
    a = st.read(i, rows=slice(0, None, 50)).astype(np.float64)
    flat = np.all(a == a[:, :1], axis=1)  # rows of one value (not scanned)
    first_flat = int(np.argmax(flat)) * 50 if flat.any() else None
    return f"mean {a.mean():8.0f} sd {a.std():7.0f} flat rows {flat.mean():5.1%} (from row {first_flat})"


def show(day, esb_idx, inl_idx):
    for t in TILES:
        e = ImageJStack(RAW / "Backscatter" / f"{day}_EsB_tile{t}.tif")
        n = ImageJStack(RAW / f"{day}_tile{t}.tif")
        print(f" tile {t}")
        for i in esb_idx:
            print(f"   EsB    [{i}] {e.labels[i][-40:]:>40} {stats(e, i)}")
        for i in inl_idx:
            print(f"   InLens [{i}] {n.labels[i][-40:]:>40} {stats(n, i)}")


print("== M10_D24: EsB has 578 slices, InLens 577 (EsB slice 404 = 15:57:27 not in InLens) ==")
show("M10_D24", [402, 403, 404, 405], [402, 403, 404])
print("\n== M11_D02: EsB 648, InLens 647 (EsB slice 239 = 08:48:18 not in InLens) ==")
show("M11_D02", [237, 238, 239, 240], [237, 238, 239])
print("\n== M11_D10 slice 583 (InLens: stopped after tile 0-0) ==")
show("M11_D10", [582, 583, 584], [582, 583, 584])

print("\n== denser copy check: md5 of 256 KB mid-slice at 8 slices per file, all files ==")
files = sorted(os.path.relpath(os.path.join(r, n), RAW) for r, d, ns in os.walk(RAW)
               if "uCT datasets" not in r for n in ns if NAME.match(n))


def samples(rel):
    st = ImageJStack(RAW / rel)
    length = min(262144, st.slice_bytes // 2)
    off = (st.slice_bytes - length) // 2
    out = []
    for i in sorted(set(np.linspace(0, st.n - 1, 8).round().astype(int))):
        b = st.read_raw_bytes(int(i) * st.slice_bytes + off, length)
        k = st.dtype.itemsize
        out.append((int(i), hashlib.md5(b).hexdigest() if b[k:] != b[:-k] else None))
    return rel, out


by, blank, n = defaultdict(list), [], 0
with ThreadPoolExecutor(8) as pool:
    for rel, out in pool.map(samples, files):
        for i, h in out:
            n += 1
            (blank.append(f"{rel}[{i}]") if h is None else by[h].append(f"{rel}[{i}]"))
shared = [v for v in by.values() if len(v) > 1]
print(f"{len(files)} files, {n} samples; shared by >1 file/slice:", "none" if not shared else "")
for v in shared:
    print("  ", ", ".join(v))
print("blank samples:", blank or "none")
