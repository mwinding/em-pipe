"""A neuroglancer link to served volumes as toggle layers (only the first one visible), opening at a given place.

    python tools/view_link.py [--port 8000] [--voxel-m X Y Z] [--position X Y Z] [--layout 4panel] [--out FILE]
                              NAME=VOLUME [NAME=VOLUME ...]

VOLUME is a volume's folder name as pipeline.serve serves it (``*.zarr`` or ``*.precomputed``), NAME the layer's
name. Sources are pipeline.serve's (``zarr3://http://localhost:PORT/VOLUME/``); the dimensions are declared in x, y, z
order (the volumes' axes are z, y, x: see pipeline.serve.neuroglancer_link) with --voxel-m the voxel size in metres,
as the volumes' OME metadata gives it; --position is in voxels (x y z).

When the neuroglancer package is installed, the link is read back with ``neuroglancer.parse_url`` and checked:
the same layers (names, sources, visibility), dimensions, position and layout, else it exits with an error. Prints
the link, and writes it to --out.
"""
import argparse
import json
import sys
from pathlib import Path
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.serve import source  # noqa: E402

PREFIX = "https://neuroglancer-demo.appspot.com/#!"


def state(layers, port=8000, voxel_m=(8e-9, 8e-9, 8e-9), position=None, layout="xy"):
    """Neuroglancer state with one image layer per (name, volume), only the first visible."""
    s = {"dimensions": {a: [float(v), "m"] for a, v in zip("xyz", voxel_m)},
         "displayDimensions": ["x", "y", "z"],
         "layers": [{"type": "image", "source": source(port, volume), "name": name, "visible": i == 0}
                    for i, (name, volume) in enumerate(layers)],
         "layout": layout}
    if position is not None:
        s["position"] = [float(v) for v in position]
    return s


def link(s):
    return PREFIX + quote(json.dumps(s, separators=(",", ":")), safe=":/,")


def check(url, s):
    """Read ``url`` back with neuroglancer.parse_url and compare it with state ``s``: a list of differences
    (empty when it matches), or None when neuroglancer is not installed."""
    try:
        import neuroglancer
    except ImportError:
        return None
    v = neuroglancer.parse_url(url)
    diffs = []
    got = [(m.name, [d.url for d in m.layer.source], bool(m.visible)) for m in v.layers]
    want = [(lay["name"], [lay["source"]], lay["visible"]) for lay in s["layers"]]
    if got != want:
        diffs.append(f"layers {got} != {want}")
    dims = v.dimensions.to_json()
    for a, (size, unit) in s["dimensions"].items():
        if a not in dims or dims[a][1] != unit or abs(float(dims[a][0]) - size) > 1e-6 * size:
            diffs.append(f"dimension {a}: {dims.get(a)} != {[size, unit]}")
    if "position" in s and [float(p) for p in v.position] != s["position"]:
        diffs.append(f"position {list(v.position)} != {s['position']}")
    layout = v.layout.type if hasattr(v.layout, "type") else v.layout
    if layout != s["layout"]:
        diffs.append(f"layout {layout!r} != {s['layout']!r}")
    return diffs


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("layers", nargs="+", help="NAME=VOLUME: layer name and served folder name")
    p.add_argument("--port", type=int, default=8000, help="the port pipeline.serve serves on")
    p.add_argument("--voxel-m", type=float, nargs=3, default=(8e-9, 8e-9, 8e-9), metavar=("X", "Y", "Z"),
                   help="voxel size in metres (x y z)")
    p.add_argument("--position", type=float, nargs=3, metavar=("X", "Y", "Z"), help="where the view opens (voxels)")
    p.add_argument("--layout", default="xy", help="xy, xz, yz, 3d, 4panel, ...")
    p.add_argument("--out", type=Path, help="also write the link to this file")
    args = p.parse_args(argv)
    layers = []
    for spec in args.layers:
        name, _, volume = spec.partition("=")
        if not name or not volume:
            p.error(f"expected NAME=VOLUME, got {spec!r}")
        layers.append((name, volume.strip("/")))
    s = state(layers, args.port, args.voxel_m, args.position, args.layout)
    url = link(s)
    diffs = check(url, s)
    if diffs:
        sys.exit("neuroglancer reads the link differently: " + "; ".join(diffs))
    print("checked with neuroglancer.parse_url" if diffs is not None else
          "neuroglancer not installed: link not read back", file=sys.stderr)
    if args.out:
        args.out.write_text(url + "\n")
    print(url)


if __name__ == "__main__":
    main()
