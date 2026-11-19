# Extracted from a session transcript (inline script, not saved as a file at the time).
# Time (UTC): 2026-10-08T20:33:21.107Z
# Ran: local
# Heredoc line: mkdir -p ${DEV_WORK} && cat > ${DEV_WORK}
# Assistant note before it: I'll use the `sleap165` env (newest tifffile). Now let me read just the header/first IFD of your example file — no pixel data, so this stays cheap over SMB.

import sys, tifffile
path = sys.argv[1]
with tifffile.TiffFile(path) as tif:
    print("file:", path)
    print("byteorder:", tif.byteorder, "bigtiff:", tif.is_bigtiff)
    print("flags:", sorted(f for f in dir(tif) if f.startswith("is_") and getattr(tif, f, False) is True))
    p = tif.pages.first
    print("page0 shape:", p.shape, "dtype:", p.dtype, "compression:", p.compression.name,
          "photometric:", p.photometric.name, "planar:", p.planarconfig.name)
    print("tiled:", p.is_tiled, "tile:", (p.tilelength, p.tilewidth) if p.is_tiled else None,
          "rowsperstrip:", p.rowsperstrip, "n_segments:", len(p.dataoffsets))
    print("page0 data offset[0]:", p.dataoffsets[0], "bytecount total:", sum(p.databytecounts))
    for tag in p.tags.values():
        v = tag.value
        s = repr(v)
        if len(s) > 600: s = s[:600] + "...<truncated>"
        print(f"  tag {tag.code} {tag.name}: {s}")
    if tif.is_imagej:
        print("imagej_metadata:", tif.imagej_metadata)
    # peek at second IFD offset without walking the whole chain
    try:
        p1 = tif.pages[1]
        print("page1 shape:", p1.shape, "offset:", p1.offset, "data offset:", p1.dataoffsets[0])
    except Exception as e:
        print("page1 err:", e)
