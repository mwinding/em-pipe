"""Train the reference tissue classifier (pipeline/refmask.py) from labelled boxes, and draw the boxes.

    python tools/refmask_train.py table LABELS.json OUT.npz --dataset NAME=CONFIG,VOLUME [--dataset ...]
    python tools/refmask_train.py fit OUT.npz MODEL.json [--lodo REPORT.json]
    python tools/refmask_train.py draw LABELS.json OUT_DIR --dataset NAME=CONFIG,VOLUME [--dataset ...]

The label file names datasets by short names; --dataset maps each to a config and a rendered volume
(a folder in the config's output_dir). --remap OLD=NEW replaces the prefix OLD of the configs'
output_dir (storage mounted elsewhere, e.g. on a laptop).

table: for every labelled plane, the cell features of its preview thumbnails (as the mask computes
  them) and, per tissue cell whose centre lies in a box, a row: features, label (1 = the reference
  tissue), dataset. The thumbnails are placed in the volume with render's frozen plan
  (work/render/<stem>/), as in refmask_check.
fit: boosted trees (pipeline/boost.py; deterministic) on the table, each (dataset, label) group given
  the same total weight; writes the model JSON (pipeline/refmask_<tissue>.json is the shipped one).
  --lodo: also leave each dataset out in turn (train on the others, test on it): per-cell AUC, and
  the fractions of its reference-tissue and other cells above probability 0.5.
draw: one PNG per labelled plane (labels_<dataset>_z<z>.png), at 128 nm (scale 4 of the volume): boxes
  in green (reference tissue) or red (other). With --previous OLD.json, boxes not in OLD are drawn in
  cyan (reference tissue) or orange (other), and OLD's boxes that are gone as thin grey outlines.

Boxes of one plane must not overlap (each cell is labelled at most once): table and draw refuse them.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from refmask_check import Dataset  # noqa: E402

from pipeline import background, boost, omezarr, preview, refmask  # noqa: E402
from pipeline.config import load_config  # noqa: E402


def load(configs, remap=None):
    """Configs, with output_dir prefix OLD replaced by NEW for remap "OLD=NEW" (storage mounted elsewhere)."""
    cfgs = [load_config(c) for c in configs]
    if remap:
        old, new = remap.split("=", 1)
        for c in cfgs:
            if c["output_dir"].startswith(old):
                c["output_dir"] = new + c["output_dir"][len(old):]
    return cfgs


def datasets(labels, specs, remap=None):
    """{name: Dataset} for every dataset named in the labels, from NAME=CONFIG,VOLUME specs."""
    given = {}
    for spec in specs:
        name, rest = spec.split("=", 1)
        config, volume = rest.split(",", 1)
        given[name] = (config, volume)
    out = {}
    for name in sorted({b["dataset"] for b in labels["boxes"]}):
        if name not in given:
            raise SystemExit(f"labels name dataset {name!r}: give --dataset {name}=CONFIG,VOLUME")
        config, volume = given[name]
        out[name] = Dataset(load([config], remap)[0], volume)
    return out


def check_overlaps(labels):
    """Raise if two boxes of one (dataset, z) overlap."""
    by = {}
    for i, b in enumerate(labels["boxes"]):
        by.setdefault((b["dataset"], b["z"]), []).append((i, b))
    bad = []
    for (d, z), bs in by.items():
        for k, (i, a) in enumerate(bs):
            for j, b in bs[k + 1:]:
                if min(a["x1"], b["x1"]) > max(a["x0"], b["x0"]) and min(a["y1"], b["y1"]) > max(a["y0"], b["y0"]):
                    bad.append(f"{d} z {z}: boxes {i} and {j}")
    if bad:
        raise SystemExit(f"{len(bad)} overlapping label boxes (each cell must be labelled once), e.g. " + "; ".join(bad[:5]))


def plane_rows(ds, z, boxes, settings):
    """Feature rows and labels of the tissue cells of plane z whose centres lie in the boxes."""
    rows = ds.index[ds.index.z == z]
    X, y, names = [], [], None
    c = settings["cell_px"]
    for r in rows.itertuples():
        thumb = np.asarray(np.load(ds.thumbs_dir / r.npy, mmap_mode="r")[int(r.i)])
        f, tissue = refmask.cell_features(preview.thumb_values(thumb, r.p0_5, r.p99_5), background.mask(thumb), c,
                                          settings["max_background"])
        names = list(f)
        M = ds.to_volume(z, r.tile, thumb.shape, 1)   # thumbnail px -> stored voxels
        cy, cx = np.mgrid[:tissue.shape[0], :tissue.shape[1]]
        px, py = cx * c + (c - 1) / 2, cy * c + (c - 1) / 2
        X_, Y_ = M[0, 0] * px + M[0, 1] * py + M[0, 2], M[1, 0] * px + M[1, 1] * py + M[1, 2]
        for b in boxes:
            sel = tissue & (X_ >= b["x0"]) & (X_ < b["x1"]) & (Y_ >= b["y0"]) & (Y_ < b["y1"])
            if sel.any():
                X.append(np.stack([f[n][sel] for n in names], axis=1))
                y.append(np.full(int(sel.sum()), b["label"] == "neuropil" or b["label"] == "reference"))
    if not X:
        return np.zeros((0, 0)), np.zeros(0, bool), names
    return np.concatenate(X), np.concatenate(y), names


def table(a):
    labels = json.loads(Path(a.labels).read_text())
    check_overlaps(labels)
    settings = {**refmask.DEFAULTS["refmask"], "tissue": labels.get("tissue", "neuropil")}
    dss = datasets(labels, a.dataset, a.remap)
    X, y, vol, zs, names = [], [], [], [], None
    for v, ds in dss.items():
        for z in sorted({b["z"] for b in labels["boxes"] if b["dataset"] == v}):
            bx = [b for b in labels["boxes"] if b["dataset"] == v and b["z"] == z]
            Xz, yz, names = plane_rows(ds, z, bx, settings)
            print(f"{v} z {z}: {len(yz)} cells ({int(yz.sum())} reference tissue)", flush=True)
            if len(yz):
                X.append(Xz)
                y.append(yz)
                vol += [v] * len(yz)
                zs += [z] * len(yz)
    np.savez_compressed(a.out, X=np.concatenate(X), y=np.concatenate(y), dataset=np.array(vol), z=np.array(zs),
                        names=np.array(names), tissue=settings["tissue"])
    return 0


def weights(y, vol):
    """Each (dataset, label) group the same total weight; mean weight 1."""
    w = np.zeros(len(y))
    groups = {(v, t) for v, t in zip(vol, y)}
    for v, t in groups:
        sel = (vol == v) & (y == t)
        w[sel] = 1.0 / sel.sum()
    return w * len(y) / w.sum()


def auc(p, y):
    r = np.argsort(np.argsort(p)) + 1
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))


def fit(a):
    d = np.load(a.table)
    X, y, vol, names = np.nan_to_num(d["X"]), d["y"].astype(bool), d["dataset"], [str(n) for n in d["names"]]
    model = boost.fit(X, y, weights(y, vol), names=names)
    model["tissue"] = str(d["tissue"])
    model["trained_on"] = sorted(set(vol.tolist()))
    model["cells"] = {"reference": int(y.sum()), "other": int((~y).sum())}
    Path(a.model).write_text(json.dumps(model))
    print(f"model: {len(model['trees'])} trees, {len(names)} features, {y.sum()} / {(~y).sum()} cells -> {a.model}")
    if a.lodo:
        rep = {}
        for v in sorted(set(vol.tolist())):
            tr, te = vol != v, vol == v
            m = boost.fit(X[tr], y[tr], weights(y[tr], vol[tr]), names=names)
            p = boost.predict(m, X[te])
            yt = y[te]
            rep[v] = {"cells_reference": int(yt.sum()), "cells_other": int((~yt).sum()),
                      "auc": auc(p, yt) if 0 < yt.sum() < len(yt) else None,
                      "recall_reference_at_0.5": float((p[yt] > 0.5).mean()) if yt.any() else None,
                      "other_kept_at_0.5": float((p[~yt] > 0.5).mean()) if (~yt).any() else None}
            print(v, rep[v], flush=True)
        Path(a.lodo).write_text(json.dumps(rep, indent=1))
    return 0


def draw(a):
    labels = json.loads(Path(a.labels).read_text())
    check_overlaps(labels)
    old = json.loads(Path(a.previous).read_text())["boxes"] if a.previous else None
    key = lambda b: (b["dataset"], b["z"], b["x0"], b["y0"], b["x1"], b["y1"], b["label"])  # noqa: E731
    new_keys = {key(b) for b in labels["boxes"]}
    old_keys = {key(b) for b in old} if old is not None else set()
    dss = datasets(labels, a.dataset, a.remap)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    f, scale = 16, 4
    for v, ds in dss.items():
        arr = omezarr.open_scale(ds.root, scale)
        for z in sorted({b["z"] for b in labels["boxes"] if b["dataset"] == v}):
            img = np.asarray(arr[ds.planes[z] // f].read().result())
            rgb = np.repeat(img[:, :, None], 3, axis=2)
            n_changed = 0
            if old is not None:
                for b in old:
                    if b["dataset"] == v and b["z"] == z and key(b) not in new_keys:
                        cv2.rectangle(rgb, (round(b["x0"] / f), round(b["y0"] / f)),
                                      (round(b["x1"] / f) - 1, round(b["y1"] / f) - 1), (150, 150, 150), 1)
            for b in labels["boxes"]:
                if b["dataset"] != v or b["z"] != z:
                    continue
                ref = b["label"] in ("neuropil", "reference")
                changed = old is not None and key(b) not in old_keys
                n_changed += changed
                col = ((255, 255, 0) if ref else (0, 140, 255)) if changed else ((60, 200, 60) if ref else (60, 60, 255))
                cv2.rectangle(rgb, (round(b["x0"] / f), round(b["y0"] / f)), (round(b["x1"] / f) - 1, round(b["y1"] / f) - 1),
                              col, 2)
            title = f"{v} z {z}, 128 nm: green neuropil, red other"
            if old is not None:
                title += f"; changed ({n_changed}): cyan neuropil, orange other; grey: removed"
            cv2.putText(rgb, title, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(rgb, title, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            name = out / f"labels_{v}_z{z}.png"
            cv2.imwrite(str(name), rgb, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            print("wrote", name, flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("table")
    t.add_argument("labels"), t.add_argument("out")
    t.add_argument("--dataset", action="append", required=True, help="NAME=CONFIG,VOLUME")
    t.add_argument("--remap", help="OLD=NEW: replace the prefix OLD of each output_dir by NEW")
    t = sub.add_parser("fit")
    t.add_argument("table"), t.add_argument("model"), t.add_argument("--lodo")
    t = sub.add_parser("draw")
    t.add_argument("labels"), t.add_argument("out_dir")
    t.add_argument("--dataset", action="append", required=True, help="NAME=CONFIG,VOLUME")
    t.add_argument("--previous", help="an earlier label file: mark what changed")
    t.add_argument("--remap", help="OLD=NEW: replace the prefix OLD of each output_dir by NEW")
    a = ap.parse_args(argv)
    return {"table": table, "fit": fit, "draw": draw}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
