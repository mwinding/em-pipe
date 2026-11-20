# Extracted from a session transcript (inline script, not saved as a file at the time).
# Time (UTC): 2026-10-08T20:34:00.081Z
# Ran: local
# Heredoc line: cat > $SP/survey.py <<'EOF'
# Assistant note before it: I'll use the `sleap165` env (newest tifffile). Now let me read just the header/first IFD of your example file — no pixel data, so this stays cheap over SMB.

import sys, os, re, json, glob, tifffile
root = sys.argv[1]; out = sys.argv[2]
files = sorted(glob.glob(os.path.join(root, "M*_D*_tile*.tif")) + glob.glob(os.path.join(root, "Approach", "M*_D*_tile*.tif")))
rows = {}
for f in files:
    with tifffile.TiffFile(f) as tif:
        p = tif.pages.first
        md = tif.imagej_metadata or {}
        labels = md.get("Labels", [])
        xres = p.tags["XResolution"].value
        rows[os.path.relpath(f, root)] = dict(
            shape=list(p.shape), dtype=str(p.dtype), n=md.get("images"), bigtiff=tif.is_bigtiff,
            byteorder=tif.byteorder, z_um=md.get("spacing"), xy_um=xres[1]/xres[0], unit=md.get("unit"),
            min=md.get("min"), max=md.get("max"), header=p.dataoffsets[0],
            size=os.path.getsize(f), labels=labels)
        r = rows[os.path.relpath(f, root)]
        expect = r["header"] + r["n"] * p.shape[0] * p.shape[1] * p.dtype.itemsize
        print(f"{os.path.relpath(f, root):38s} {p.shape[0]:>6}x{p.shape[1]:<6} n={r['n']:<4} "
              f"z={r['z_um']*1000:.3f}nm xy={r['xy_um']*1000:.3f}nm size_ok={expect==r['size']} "
              f"first={labels[0] if labels else None} last={labels[-1] if labels else None}", flush=True)
json.dump(rows, open(out, "w"))
