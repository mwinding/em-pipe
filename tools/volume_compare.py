"""Same-place views of two or more rendered volumes side by side, with grey histograms: e.g. a re-render
against an earlier delivery, or one volume against another's settings.

Sources: a local OME-Zarr (``.ome.zarr``, as render writes) or a Neuroglancer precomputed volume (a
``.precomputed`` folder, or a ``gs://bucket/prefix`` URL, as the precomputed export and upload write). The
export has one scale per zarr scale, (x, y, z) axes and no offset, so scale k and voxel (z, y, x) are the same
place in both. Bucket sources are read by tensorstore with Google's application default credentials (e.g.
GOOGLE_APPLICATION_CREDENTIALS set to a key file's path); this tool never opens the key itself.

For every z: the whole slice at ``overview_scale`` and full-resolution crops of ``crop_px`` square, one column
per volume, each row with the histogram of its panels' greys (no-data 0 left out) and their p5 / median / p95.
Crops are given per z as (y, x) centres at full resolution, or picked: textured tissue in the first volume's
overview (cell bodies and their borders more than plain neuropil), spread over the slice.

Spec (JSON): {"volumes": [{"label": "...", "source": "..."}, ...], "z": [1000, 5760],
              "overview_scale": 5, "crop_px": 768, "crops_per_z": 2, "crops": {"5760": [[y, x], ...]},
              "seed": 0}

  python tools/volume_compare.py SPEC.json OUT.png [--json OUT.json]      # writes OUT_z<z>.png per z
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorstore as ts
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import omezarr  # noqa: E402

COLOURS = ["#3b6fd4", "#d9822b", "#2e9e6b", "#a24db8"]


class Volume:
    """One source, read as (z, y, x) at any scale."""

    def __init__(self, source):
        self.source = str(source).rstrip("/")
        self.zarr = self.source.endswith(".zarr")
        self._scales = {}

    def scale(self, k):
        if k not in self._scales:
            if self.zarr:
                self._scales[k] = omezarr.open_scale(self.source, k)
            else:
                kv = self.source + "/" if "://" in self.source else {"driver": "file", "path": self.source + "/"}
                self._scales[k] = ts.open({"driver": "neuroglancer_precomputed", "kvstore": kv, "scale_index": k},
                                          open=True, read=True).result()
        return self._scales[k]

    def shape(self, k=0):
        s = [int(v) for v in self.scale(k).shape]
        return tuple(s) if self.zarr else (s[2], s[1], s[0])

    def plane(self, k, z, y0=0, y1=None, x0=0, x1=None):
        """Slice z (at scale k's own z) as a (y, x) array."""
        a = self.scale(k)
        if self.zarr:
            return np.asarray(a[z, y0:y1, x0:x1].read().result())
        return np.asarray(a[x0:x1, y0:y1, z, 0].read().result()).T


def stats(img):
    """p5 / median / p95 of the greys, and the share at 255, leaving no-data 0 out."""
    g = img[img > 0]
    if g.size == 0:
        return {"n": 0}
    p5, p50, p95 = np.percentile(g, [5, 50, 95])
    return {"n": int(g.size), "p5": float(p5), "median": float(p50), "p95": float(p95),
            "width": float(p95 - p5), "at_255_pct": float(100 * np.mean(g == 255))}


def pick_crops(overview, f, crop_px, n, rng):
    """(y, x) centres at full resolution of up to n crops in textured tissue of an overview at downsample f:
    windows wholly inside the data, local SD above the median of those, at least a quarter slice apart."""
    w = max(1, round(crop_px / f))
    img = overview.astype(float)
    inside = ndimage.minimum_filter(overview > 0, size=w)
    m = ndimage.uniform_filter(img, w)
    sd = np.sqrt(np.maximum(ndimage.uniform_filter(img * img, w) - m * m, 0))
    if not inside.any():
        return []
    cand = np.argwhere(inside & (sd >= np.median(sd[inside])))
    rng.shuffle(cand)
    gap = min(overview.shape) / 4
    picks = []
    for c in cand:
        if all(np.hypot(*(c - p)) >= gap for p in picks):
            picks.append(c)
            if len(picks) == n:
                break
    return [[int(round((y + 0.5) * f)), int(round((x + 0.5) * f))] for y, x in picks]


def _row(fig, gs, r, panels, vols, title, interp):
    out = {}
    hist = fig.add_subplot(gs[r, len(vols)])
    for i, (v, img) in enumerate(zip(vols, panels)):
        ax = fig.add_subplot(gs[r, i])
        ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation=interp)
        ax.set_xticks([])
        ax.set_yticks([])
        st = stats(img)
        out[v["label"]] = st
        txt = f"{v['label']}\n" + (f"p5 {st['p5']:.0f} · median {st['median']:.0f} · p95 {st['p95']:.0f}"
                                    if st["n"] else "no data")
        ax.set_title(txt, fontsize=10)
        g = img[img > 0]
        if g.size:
            h = np.bincount(g.ravel(), minlength=256)[1:] / g.size
            hist.plot(np.arange(1, 256), h, color=COLOURS[i % len(COLOURS)], lw=1.5, label=v["label"])
    hist.set_xlim(0, 255)
    hist.set_ylim(bottom=0)
    hist.set_xlabel("grey")
    hist.set_yticks([])
    hist.set_title(title, fontsize=10)
    hist.legend(fontsize=8, frameon=False, loc="upper left")
    for s in ("top", "right", "left"):
        hist.spines[s].set_visible(False)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("spec", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--json", type=Path)
    args = p.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spec = json.loads(args.spec.read_text())
    vols = spec["volumes"]
    srcs = [Volume(v["source"]) for v in vols]
    shapes = [s.shape(0) for s in srcs]
    for v, sh in zip(vols, shapes):
        print(f"{v['label']}: shape (z, y, x) {sh}", flush=True)
    if len(set(shapes)) > 1:
        sys.exit("the volumes differ in shape: not the same geometry, crops would not match")
    k = int(spec.get("overview_scale", 5))
    f = 2 ** k
    crop = int(spec.get("crop_px", 768))
    rng = np.random.default_rng(spec.get("seed", 0))
    result = {}
    for z in spec["z"]:
        overviews = [s.plane(k, z // f) for s in srcs]
        yx = spec.get("crops", {}).get(str(z)) or pick_crops(overviews[0], f, crop, int(spec.get("crops_per_z", 2)), rng)
        rows = 1 + len(yx)
        fig = plt.figure(figsize=(5.5 * len(vols) + 4.5, 5.8 * rows))
        gs = fig.add_gridspec(rows, len(vols) + 1, width_ratios=[1] * len(vols) + [0.8], hspace=0.25, wspace=0.05)
        res = {"overview": _row(fig, gs, 0, overviews, vols, f"z {z}: whole slice ({8 * f} nm pixels)", "antialiased"),
               "crops": []}
        for r, (y, x) in enumerate(yx, start=1):
            _, ny, nx = shapes[0]
            y0 = min(max(0, y - crop // 2), max(0, ny - crop))
            x0 = min(max(0, x - crop // 2), max(0, nx - crop))
            panels = [s.plane(0, z, y0, min(ny, y0 + crop), x0, min(nx, x0 + crop)) for s in srcs]
            st = _row(fig, gs, r, panels, vols, f"crop at y {y}, x {x} ({crop * 8 / 1000:.1f} µm, 8 nm pixels)",
                      "nearest")
            res["crops"].append({"yx": [y, x], **st})
        path = args.out.with_name(f"{args.out.stem}_z{z}.png")
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {path}", flush=True)
        result[str(z)] = res
    if args.json:
        args.json.write_text(json.dumps({"volumes": [v["label"] for v in vols], "z": result}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
