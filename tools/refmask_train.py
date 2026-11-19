"""Train the reference tissue classifier (pipeline/refmask.py) from labelled boxes, and draw the boxes.

    python tools/refmask_train.py table LABELS.json OUT.npz --dataset NAME=CONFIG,VOLUME [--dataset ...]
    python tools/refmask_train.py fit OUT.npz MODEL.json [--lodo REPORT.json]
    python tools/refmask_train.py draw LABELS.json OUT_DIR --dataset NAME=CONFIG,VOLUME [--dataset ...]
    python tools/refmask_train.py propose LABELS.json OUT.json --dataset ... --planes NAME:Z,...

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
  cyan (reference tissue) or orange (other), and OLD's boxes that are gone as dashed magenta outlines.
  --outline: the classifier's mask outline as a thin yellow line. Box IDs are written next to the
  boxes, and OUT_DIR/boxes.csv lists every box (dataset, z, id, label, x0, y0, x1, y1, note).

propose: conservative boxes from the current classifier's mask. A box (--box-um, 10 um) is a
  reference-tissue box only if it and a margin of --margin-um (10 um) all round lie inside the mask in
  every tile covering it, and an "other" box only if box and margin are tissue with no mask, no
  candidate cell (probability over the threshold) and no background. Boxes already in LABELS.json
  are kept only if they pass the same test (dropped ones are listed). On each --planes plane up to
  --per-plane boxes of each label are added, spread out (farthest-point, at least --spacing-um apart
  and not overlapping kept boxes). Every box gets an ID per plane: N1, N2... (reference tissue),
  C1, C2... (other); boxes kept from LABELS.json keep their IDs. No box is proposed over (box plus
  margin) a box listed in LABELS.json's "rejected_in_review". Drawing positives from the classifier's
  own mask is self-referential: only a person's check of the pictures (draw --outline) validates
  them. "other" boxes must be cell bodies only, never gut, oesophagus or trachea: propose can't tell
  those apart from cell bodies (they also lack the mask), so the person checking rejects them.

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


def plane_regions(ds, z, settings, f=16):
    """At downsample f of the stored volume: (reference_ok, other_ok, mask) of plane z.

    reference_ok: inside the mask in every tile covering the pixel; other_ok: tissue with neither
    mask nor candidate (probability or score over the threshold) in every covering tile; mask: the
    mask in any tile."""
    rows = ds.index[ds.index.z == z]
    tiles = {r.tile: (np.asarray(np.load(ds.thumbs_dir / r.npy, mmap_mode="r")[int(r.i)]), r.p0_5, r.p99_5)
             for r in rows.itertuples()}
    masks, scores, thr = refmask.slice_masks(tiles, settings)
    W, H = (int(np.ceil(v / f)) for v in ds.meta["canvas_size_xy"])
    covered, mask_any = np.zeros((H, W), bool), np.zeros((H, W), bool)
    ref_ok, other_ok = np.ones((H, W), bool), np.ones((H, W), bool)
    c = settings["cell_px"]
    for t, (thumb, _, _) in tiles.items():
        sc = scores[t]
        parts = [refmask.to_pixels(a, thumb.shape, c) for a in
                 (masks[t], np.nan_to_num(sc, nan=-np.inf) > thr, np.isfinite(sc))]
        M = ds.to_volume(z, t, thumb.shape, f)
        m, cand, tis, cov = (cv2.warpAffine(a.astype(np.uint8), M, (W, H), flags=cv2.INTER_NEAREST) > 0
                             for a in (*parts, np.ones(thumb.shape, bool)))
        covered |= cov
        mask_any |= m
        ref_ok &= ~cov | m
        other_ok &= ~cov | (tis & ~cand & ~m)
    return ref_ok & covered, other_ok & covered, mask_any


def _inside(ok, box, margin, f):
    """True if box (stored voxels) plus margin lies wholly in ok (at downsample f)."""
    y0, x0 = int(np.floor((box["y0"] - margin) / f)), int(np.floor((box["x0"] - margin) / f))
    y1, x1 = int(np.ceil((box["y1"] + margin) / f)), int(np.ceil((box["x1"] + margin) / f))
    if y0 < 0 or x0 < 0 or y1 > ok.shape[0] or x1 > ok.shape[1]:
        return False
    return bool(ok[y0:y1, x0:x1].all())


def _spread(ok, b, m, n, spacing, existing, step=4):
    """Up to n top-left corners (downsample px) of b-px boxes whose box plus m px margin lies in ok,
    farthest-point spread, at least spacing px between box centres and from the existing centres."""
    w = b + 2 * m
    S = np.pad(np.cumsum(np.cumsum(ok.astype(np.int64), 0), 1), ((1, 0), (1, 0)))
    if ok.shape[0] < w or ok.shape[1] < w:
        return []
    cnt = S[w:, w:] - S[:-w, w:] - S[w:, :-w] + S[:-w, :-w]
    ys, xs = np.nonzero(cnt[::step, ::step] == w * w)
    if not len(ys):
        return []
    cy, cx = ys * step + m + b / 2, xs * step + m + b / 2
    chosen = [np.asarray(e, float) for e in existing]
    picked = []
    d = np.full(len(cy), np.inf)
    for e in chosen:
        d = np.minimum(d, np.hypot(cy - e[0], cx - e[1]))
    if not chosen:   # start nearest the middle of the allowed area
        i = int(np.argmin(np.hypot(cy - cy.mean(), cx - cx.mean())))
    else:
        i = int(np.argmax(d))
    while len(picked) < n and d[i] >= spacing:
        picked.append((int(ys[i] * step + m), int(xs[i] * step + m)))
        d = np.minimum(d, np.hypot(cy - cy[i], cx - cx[i]))
        i = int(np.argmax(d))
    return picked


def propose(a):
    labels = json.loads(Path(a.labels).read_text())
    settings = {**refmask.DEFAULTS["refmask"], "tissue": labels.get("tissue", "neuropil")}
    planes = [(p.split(":")[0], int(p.split(":")[1])) for p in a.planes.split(",")] if a.planes else []
    names = {d.split("=", 1)[0] for d in a.dataset}
    if any(d not in names for d, _ in planes):
        raise SystemExit("every --planes dataset needs a --dataset")
    labels = {**labels, "boxes": [b for b in labels["boxes"] if b["dataset"] in names]}
    dss = datasets({"boxes": labels["boxes"] + [{"dataset": d} for d, _ in planes]}, a.dataset, a.remap)
    f = 16
    box, margin, spacing = (round(v * 1000 / 8) for v in (a.box_um, a.margin_um, a.spacing_um))   # 8 nm voxels
    todo = sorted({(b["dataset"], b["z"]) for b in labels["boxes"]} | set(planes))
    out, dropped = [], []
    for d, z in todo:
        ds = dss[d]
        ref_ok, other_ok, _ = plane_regions(ds, z, settings, f)
        for r in labels.get("rejected_in_review", []):   # nothing new over a rejected box
            if (r["dataset"], r["z"]) == (d, z):
                y0, x0 = max((r["y0"] - margin - box) // f, 0), max((r["x0"] - margin - box) // f, 0)
                y1, x1 = (r["y1"] + margin + box) // f + 1, (r["x1"] + margin + box) // f + 1
                ref_ok[y0:y1, x0:x1] = False
                other_ok[y0:y1, x0:x1] = False
        kept = {"ref": [], "other": []}
        for b in labels["boxes"]:
            if (b["dataset"], b["z"]) != (d, z):
                continue
            ref = b["label"] in ("neuropil", "reference")
            if _inside(ref_ok if ref else other_ok, b, margin, f):
                kept["ref" if ref else "other"].append(dict(b))
            else:
                dropped.append(b)
        if (d, z) in planes:
            for kind, ok, lab in (("ref", ref_ok, settings["tissue"]), ("other", other_ok, "other")):
                existing = [((b["y0"] + b["y1"]) / 2 / f, (b["x0"] + b["x1"]) / 2 / f) for b in kept[kind]]
                n = max(a.per_plane - len(existing), 0)
                for y, x in _spread(ok, box // f, margin // f, n, spacing / f, existing):
                    kept[kind].append({"dataset": d, "z": z, "x0": x * f, "y0": y * f, "x1": x * f + box, "y1": y * f + box,
                                       "label": lab, "note": "proposed from the classifier mask (box and margin inside)"})
        for kind, prefix in (("ref", "N"), ("other", "C")):
            used = [int(b["id"][1:]) for b in kept[kind] if str(b.get("id", "")).startswith(prefix)]
            nxt = max(used, default=0)
            for b in sorted(kept[kind], key=lambda b: (b["y0"], b["x0"])):
                if not str(b.get("id", "")).startswith(prefix):
                    nxt += 1
                    b["id"] = f"{prefix}{nxt}"
                out.append(b)
        print(f"{d} z {z}: {sum(b['label'] != 'other' for b in out if (b['dataset'], b['z']) == (d, z))} reference, "
              f"{sum(b['label'] == 'other' for b in out if (b['dataset'], b['z']) == (d, z))} other", flush=True)
    res = {k: v for k, v in labels.items() if k != "boxes"}
    res["description"] = (f"Labelled boxes for the reference tissue classifier, proposed by tools/refmask_train.py propose: "
                          f"box {a.box_um:g} um, margin {a.margin_um:g} um all round inside the classifier's mask (reference "
                          f"tissue) or inside tissue with no mask, candidate or background (other). Drawn from the classifier's own "
                          f"mask, so only a person's check of the pictures validates them. 'dataset' names map to a config and "
                          f"volume (--dataset NAME=CONFIG,VOLUME); boxes in that volume's stored voxels (x0, y0, x1, y1; x1/y1 "
                          f"exclusive), plane z; 'id' is per plane. Boxes of one plane must not overlap.")
    res["boxes"] = out
    if labels.get("rejected_in_review"):
        res["rejected_in_review"] = labels["rejected_in_review"]
    check_overlaps(res)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"{len(out)} boxes -> {a.out}; dropped {len(dropped)} input boxes:", flush=True)
    for b in dropped:
        print(f"  dropped {b['dataset']} z {b['z']} {b['label']} x {b['x0']}-{b['x1']} y {b['y0']}-{b['y1']} ({b.get('note', '')})")
    return 0


def dashed_rect(img, x0, y0, x1, y1, col, dash=8):
    for x in range(x0, x1, 2 * dash):
        cv2.line(img, (x, y0), (min(x + dash, x1), y0), col, 1)
        cv2.line(img, (x, y1), (min(x + dash, x1), y1), col, 1)
    for y in range(y0, y1, 2 * dash):
        cv2.line(img, (x0, y), (x0, min(y + dash, y1)), col, 1)
        cv2.line(img, (x1, y), (x1, min(y + dash, y1)), col, 1)


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
    settings = {**refmask.DEFAULTS["refmask"], "tissue": labels.get("tissue", "neuropil")}
    rows_csv = ["dataset,z,id,label,x0,y0,x1,y1,note"]
    for v, ds in dss.items():
        arr = omezarr.open_scale(ds.root, scale)
        for z in sorted({b["z"] for b in labels["boxes"] if b["dataset"] == v}):
            img = np.asarray(arr[ds.planes[z] // f].read().result())
            rgb = np.repeat(img[:, :, None], 3, axis=2)
            if a.outline:
                _, _, mk = plane_regions(ds, z, settings, f)
                mk = mk[:rgb.shape[0], :rgb.shape[1]]
                cnt, _ = cv2.findContours(mk.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(rgb, cnt, -1, (0, 230, 255), 1)
            n_changed = 0
            if old is not None:
                for b in old:
                    if b["dataset"] == v and b["z"] == z and key(b) not in new_keys:
                        dashed_rect(rgb, round(b["x0"] / f), round(b["y0"] / f), round(b["x1"] / f) - 1,
                                    round(b["y1"] / f) - 1, (255, 0, 255))
            for b in labels["boxes"]:
                if b["dataset"] != v or b["z"] != z:
                    continue
                ref = b["label"] in ("neuropil", "reference")
                changed = old is not None and key(b) not in old_keys
                n_changed += changed
                col = ((255, 255, 0) if ref else (0, 140, 255)) if changed else ((60, 200, 60) if ref else (60, 60, 255))
                p0, p1 = (round(b["x0"] / f), round(b["y0"] / f)), (round(b["x1"] / f) - 1, round(b["y1"] / f) - 1)
                cv2.rectangle(rgb, p0, p1, col, 2)
                if b.get("id"):
                    org = (p0[0] + 3, p0[1] - 6 if p0[1] > 24 else p1[1] + 20)
                    cv2.putText(rgb, b["id"], org, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 5, cv2.LINE_AA)
                    cv2.putText(rgb, b["id"], org, cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2, cv2.LINE_AA)
                note = b.get("note", "").replace(",", ";")
                rows_csv.append(f"{v},{z},{b.get('id', '')},{b['label']},{b['x0']},{b['y0']},{b['x1']},{b['y1']},{note}")
            title = f"{v} z {z}, 128 nm: green neuropil, red other" + ("; yellow: mask outline" if a.outline else "")
            if old is not None:
                title += f"; new/changed ({n_changed}): cyan neuropil, orange other; dashed magenta: removed"
            cv2.putText(rgb, title, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(rgb, title, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            name = out / f"labels_{v}_z{z}.png"
            cv2.imwrite(str(name), rgb, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            print("wrote", name, flush=True)
    (out / "boxes.csv").write_text("\n".join(rows_csv) + "\n")
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
    t.add_argument("--outline", action="store_true", help="draw the classifier's mask outline (thin yellow)")
    t.add_argument("--remap", help="OLD=NEW: replace the prefix OLD of each output_dir by NEW")
    t = sub.add_parser("propose")
    t.add_argument("labels"), t.add_argument("out")
    t.add_argument("--dataset", action="append", required=True, help="NAME=CONFIG,VOLUME")
    t.add_argument("--planes", help="NAME:Z,... planes to add boxes on")
    t.add_argument("--box-um", type=float, default=10.0)
    t.add_argument("--margin-um", type=float, default=10.0)
    t.add_argument("--spacing-um", type=float, default=30.0)
    t.add_argument("--per-plane", type=int, default=6, help="boxes per label and plane (kept ones included)")
    t.add_argument("--remap", help="OLD=NEW: replace the prefix OLD of each output_dir by NEW")
    a = ap.parse_args(argv)
    return {"table": table, "fit": fit, "draw": draw, "propose": propose}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
