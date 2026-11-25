"""A one-page QC summary of a rendered volume: one self-contained HTML file.

    python tools/qc_summary.py CONFIG [--volume NAME] [--out HTML] [--flicker DIR] [--disk-size]

CONFIG names the dataset and the volume (render.name; --volume picks another volume of the same
output_dir, which then shows the volume's own render settings from its render.json). The page is put
together from what the pipeline and the tools have already written under <output_dir>: nothing heavy
is measured again. A part that has not been run is listed as "not run", with the command that makes it.

Sections: dataset (slices, tiles, voxel size, segments, pauses, dates, excluded slices), processing
settings (the ones that define the volume, and the em-pipe commit that wrote the page), checks (each
with its numbers and a pass / attention mark against the limit stated next to it), slice quality flags
(tools/slice_quality.py), figures (the run's existing qc plots, downsized and embedded) and files.

The limits are working values for this page (LIMITS below), not validated standards: "attention"
means "look at this", not "the volume is bad".

--flicker DIR   the tools/zflicker.py output folder to report (default: every qc/zflicker*/ folder)
--disk-size     also add up the volume's size on disk (walks every file: slow on large volumes)

Writes <output_dir>/qc/summary_<volume>.html (or --out). Paths on the page are relative to the qc
folder; images are embedded, there are no scripts and nothing is loaded from the network.
"""
import argparse
import base64
import html
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline import render  # noqa: E402
from pipeline.cli import atomic_write  # noqa: E402
from pipeline.config import load_config, step_defaults, step_path  # noqa: E402

LIMITS = {
    "stitch_failed_samples": 0,        # stitched samples without a solution
    "stitch_residual_px": 2.0,         # median residual of the tile pairs used
    "align_rejected_share": 0.005,     # share of slice pairs the alignment rejected
    "align_rms_px": 5.0,               # median residual of the accepted slice pairs
    "big_jump_px": 15.0,               # a tile movement between two slices counted as a large jump
    "reference_in_band": 0.95,         # share of sampled slices with the neuropil within the standard's band
    "clipped_max": 0.01,               # largest predicted share of tissue clipped at 0 or 255 in a sampled slice
    "flicker_common_sd": 1.0,          # slice-wide flicker of the volume, greys (robust sd)
    "thickness_cv": 0.15,              # scatter of the estimated slice spacing / nominal
    "major_flag_share": 0.01,          # share of slices with a major slice-quality flag
}
PASS, ATTN, NOTRUN, INFO = "pass", "attention", "not run", "info"
FIGURE_WIDTH = 1400


def esc(x):
    return html.escape(str(x))


def fmt(v, digits=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    return f"{float(v):.{digits}g}"


def read_csv(path, **kw):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(path, **kw)
    except pd.errors.EmptyDataError:
        return None


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def rel(path, qc):
    """A path as text relative to the qc folder (never absolute)."""
    return os.path.relpath(str(path), str(qc)).replace(os.sep, "/")


def commit():
    try:
        r = subprocess.run(["git", "-C", str(ROOT), "describe", "--always", "--dirty", "--abbrev=10"],
                           capture_output=True, text=True, timeout=20)
        return r.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def table(head, rows, cls=""):
    if not rows:
        return ""
    h = "".join(f"<th>{esc(c)}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="scroll"><table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def mark(status):
    return f'<span class="mark {status.replace(" ", "-")}">{esc(status)}</span>'


# --- sections -----------------------------------------------------------------------------------------------

def dataset_section(cfg, R):
    sl = read_csv(step_path(cfg, "check", "slices.csv"), dtype={"tile": str, "exclude_reason": str}, keep_default_na=False)
    if sl is None:
        return "<p>" + mark(NOTRUN) + " The raw-data check has not run: <code>./run_pipeline.sh CONFIG --steps check</code></p>", {}
    for c in ("seam", "excluded"):
        sl[c] = sl[c].astype(str).str.lower().isin(["true", "1"])
    sl["timestamp"] = pd.to_datetime(sl["timestamp"])
    plan = R.get("plan")
    used = sl[~sl["excluded"]]
    if plan is not None:
        used = used[used["z"].isin(set(plan["z"]))]
    files = read_csv(step_path(cfg, "check", "files.csv"), dtype={"file": str})
    vox = None
    if files is not None and {"voxel_z_nm", "voxel_y_nm", "voxel_x_nm"} <= set(files.columns):
        vox = [pd.to_numeric(files[c], errors="coerce").median() for c in ("voxel_z_nm", "voxel_y_nm", "voxel_x_nm")]
    facts = [("Slices in the volume", fmt(used["z"].nunique()) + (f" (z {used['z'].min()}-{used['z'].max()})" if len(used) else "")),
             ("Tile images", fmt(len(used))),
             ("Acquired", f"{used['timestamp'].min():%Y-%m-%d %H:%M} to {used['timestamp'].max():%Y-%m-%d %H:%M}" if len(used) else "n/a"),
             ("Raw voxel size (z, y, x)", " x ".join(f"{v:.3g}" for v in vox) + " nm" if vox else "n/a")]
    if R.get("render"):
        r = R["render"]
        facts.append(("Volume size (z, y, x)", " x ".join(f"{v:,}" for v in r["shape"]) + " voxels"))
        facts.append(("Volume voxel size (z, y, x)", " x ".join(f"{v:.4g}" for v in r["voxel_nm"]) + " nm"))
    seg_rows = []
    for seg, g in used.groupby("segment"):
        seg_rows.append([seg, f"{g['tile_row'].nunique()} x {g['tile_col'].nunique()}", f"{int(g['height'].iloc[0])} x {int(g['width'].iloc[0])}",
                         f"{g['z'].min()}-{g['z'].max()}", fmt(g["z"].nunique()),
                         f"{g['timestamp'].min():%Y-%m-%d %H:%M} to {g['timestamp'].max():%Y-%m-%d %H:%M}"])
    seams = sorted(used.loc[used["seam"], "z"].unique().tolist())
    facts.append(("Segments (runs of one tile layout)", fmt(used["segment"].nunique())))
    facts.append(("Seams (first slice after a pause, restart or layout change)", f"{len(seams)}" + (": z " + ", ".join(map(str, seams[:40])) + (" ..." if len(seams) > 40 else "") if seams else "")))
    ex = sl[sl["excluded"]]
    ex_rows = [[esc(reason or "no reason given"), fmt(g["z"].nunique()), esc(", ".join(map(str, sorted(g["z"].unique())[:20])) + (" ..." if g["z"].nunique() > 20 else ""))]
               for reason, g in ex.groupby("exclude_reason")]
    facts.append(("Excluded slices", fmt(ex["z"].nunique()) if len(ex) else "none"))
    out = table(["", ""], [[esc(k), esc(v)] for k, v in facts], "facts")
    out += "<h3>Segments</h3>" + table(["segment", "tiles (rows x columns)", "tile (h x w px)", "z", "slices", "time"], [[esc(c) for c in r] for r in seg_rows])
    if ex_rows:
        out += "<h3>Excluded slices</h3>" + table(["reason", "slices", "z"], ex_rows)
    return out, {"n_slices": int(used["z"].nunique()), "seams": seams}


def settings_section(cfg, R):
    r = R.get("render") or {}
    rs = r.get("settings") or cfg.get("render") or {}
    ds = r.get("destreak") if r else cfg.get("destreak")
    i, s, a = cfg.get("intensity") or {}, cfg.get("stitch") or {}, cfg.get("align") or {}
    note = "" if R.get("own_config") else " (from the config given; a volume chosen with --volume may have been made with other intensity settings)"
    rows = [("Stitch", f"model {s.get('model')}, every {s.get('sample_every')}th slice, refined where tiles move more than {s.get('refine_px')} px"),
            ("Align", f"model {a.get('model')}, {a.get('neighbors')} neighbour slices; layout changes: {a.get('segment_change_model')}"
                      + (f"; elastic fields: {len(r['elastic'].get('fields', []))}" if r.get("elastic") else "")),
            ("Flat-field (shading)", "on" if (cfg.get("shading") or {}).get("enabled") else "off"),
            ("Tile balance", f"source {i.get('balance_source')}" + (f", patch grid {(cfg.get('overlaps') or {}).get('patch_grid')}" if (cfg.get("overlaps") or {}).get("enabled") else "")),
            ("Patch shapes (patchgrid)", "on" if (cfg.get("patchgrid") or {}).get("enabled") else "off"),
            ("Balance along z", f"balance_z_distance {i.get('balance_z_distance')}, per_slice {i.get('per_slice')}, slice_level {i.get('slice_level')}"),
            ("Contrast window" + note, f"{i.get('window')}" + (f": neuropil median {i['reference'].get('median')}, p5-p95 width {i['reference'].get('contrast')}, per {i['reference'].get('per')}"
                                                         if i.get("window") == "reference" and isinstance(i.get("reference"), dict) else
                                                         f" ({i.get('lo_percentile')}-{i.get('hi_percentile')} % of {i.get('percentile_of')})" if i.get("window") == "percentile" else "")),
            ("Curtain removal (destreak)", "off" if not (ds or {}).get("enabled") else
             ("swap from a second detector" if ds.get("reference_from") else "wavelet-FFT filter")
             + f", levels {ds.get('levels')}, stripe_px {ds.get('stripe_px')}, core_threshold {ds.get('core_threshold')}, "
               f"band_match {ds.get('band_match')}" + (f" (bands up to {ds.get('band_max_px')} px)" if ds.get("band_match") else "")),
            ("Slice-thickness correction (zcorrect)", "applied" if r.get("zcorrected") else "not applied"),
            ("Render", f"downsample {rs.get('downsample')}, blend {rs.get('blend')}, rotation {rs.get('rotate_deg', 0)} deg, "
                       f"{rs.get('num_scales')} scales, compression {rs.get('compression')}"),
            ("CLAHE", (f"on (clip {rs['clahe'].get('clip_limit')}, {rs['clahe'].get('tile_px')} px)" if (rs.get("clahe") or {}).get("enabled") else "off")),
            ("em-pipe commit that wrote this page", R["commit"])]
    return table(["setting", "value"], [[esc(k), esc(v)] for k, v in rows], "facts")


def checks(cfg, R, facts):
    """[(check, numbers, limit, status, detail html)]"""
    qc, out = R["qc"], []
    cmd = lambda c: f"<code>{esc(c)}</code>"   # noqa: E731
    # raw data check
    iss = read_csv(step_path(cfg, "check", "issues.csv"))
    if iss is None and not step_path(cfg, "check", "issues.csv").exists():
        out.append(("Raw data check", "", "", NOTRUN, cmd("./run_pipeline.sh CONFIG --steps check")))
    else:
        iss = iss if iss is not None else pd.DataFrame(columns=["severity", "code", "known"])
        known = iss["known"].astype(str).str.lower().isin(["true", "1"]) if len(iss) else pd.Series([], dtype=bool)
        err = iss[(iss["severity"] == "ERROR") & ~known]
        codes = iss.groupby(["severity", "code"]).size()
        txt = ", ".join(f"{sev} {code}: {n}" for (sev, code), n in codes.items()) or "no issues"
        out.append(("Raw data check", esc(txt), "no ERROR outside check.known_issues", PASS if not len(err) else ATTN,
                    f'<a href="{esc(rel(qc / "check_report.md", qc))}">check_report.md</a>'))
    # stitch
    layout = read_json(step_path(cfg, "stitch", "layout.json"))
    pairs = read_csv(step_path(cfg, "stitch", "pairs.csv"))
    if layout is None:
        out.append(("Stitch", "", "", NOTRUN, cmd("./run_pipeline.sh CONFIG --steps stitch")))
    else:
        segs = layout["segments"].values()
        failed = sum(len(s.get("failed_samples", [])) for s in segs)
        n_s = sum(s.get("n_samples", 0) for s in segs)
        med = np.nan
        if pairs is not None and len(pairs):
            used = pairs[pairs["used"].astype(str).str.lower().isin(["true", "1"])]
            col = "solve_residual_px" if "solve_residual_px" in used else "residual_px"
            med, p99 = (used[col].median(), used[col].quantile(0.99)) if len(used) else (np.nan, np.nan)
            txt = f"{fmt(n_s)} slices stitched, {failed} failed; tile-pair residual median {fmt(med)} px, 99th percentile {fmt(p99)} px"
        else:
            txt = f"{fmt(n_s)} slices stitched, {failed} failed (single tile or no pairs)"
        ok = failed <= LIMITS["stitch_failed_samples"] and not (np.isfinite(med) and med > LIMITS["stitch_residual_px"])
        out.append(("Stitch", esc(txt), f"failed = {LIMITS['stitch_failed_samples']}, median residual <= {LIMITS['stitch_residual_px']} px",
                    PASS if ok else ATTN, ""))
    # align
    res = read_csv(step_path(cfg, "align", "residuals.csv"))
    if res is None:
        out.append(("Align", "", "", NOTRUN, cmd("./run_pipeline.sh CONFIG --steps align")))
    else:
        rej = res["rejected"].astype(str).str.lower().isin(["true", "1"])
        share = float(rej.mean()) if len(res) else 0.0
        med = res.loc[~rej, "rms_px"].median()
        txt = (f"{fmt(len(res))} slice pairs, {fmt(int(rej.sum()))} rejected ({100 * share:.2f} %); residual of the accepted pairs: "
               f"median {fmt(med)} px, 99th percentile {fmt(res.loc[~rej, 'rms_px'].quantile(0.99))} px")
        ok = share <= LIMITS["align_rejected_share"] and not med > LIMITS["align_rms_px"]
        out.append(("Align", esc(txt), f"rejected <= {100 * LIMITS['align_rejected_share']:g} %, median <= {LIMITS['align_rms_px']} px",
                    PASS if ok else ATTN, ""))
    # tile jumps
    tiles = read_csv(step_path(cfg, "stitch", "tiles.csv"), dtype={"tile": str})
    if tiles is not None and tiles["tile"].nunique() > 1:
        t = tiles.sort_values(["segment", "tile", "z"])
        d = np.hypot(t.groupby(["segment", "tile"])["tx"].diff(), t.groupby(["segment", "tile"])["ty"].diff())
        step = d.groupby(t["z"]).max().dropna()
        big = step[step > LIMITS["big_jump_px"]]
        txt = (f"tile movement between consecutive slices: {int((step > 5).sum())} slices over 5 px, {len(big)} over "
               f"{LIMITS['big_jump_px']:g} px, largest {fmt(step.max() if len(step) else 0.0)} px"
               + (" (z " + ", ".join(map(str, big.sort_values(ascending=False).index[:10].tolist())) + ")" if len(big) else ""))
        has_plot = (qc / "tile_jumps.png").exists()
        out.append(("Tile jumps", esc(txt), f"listed for information; over {LIMITS['big_jump_px']:g} px: check the slices", ATTN if len(big) else PASS,
                    "" if has_plot else "plot: " + cmd("python tools/tile_jumps.py --config CONFIG")))
    # layout changes and elastic corrections
    ch = read_csv(step_path(cfg, "align", "segment_changes.csv"))
    el = read_csv(step_path(cfg, "align", "elastic_interfaces.csv"))
    if ch is not None and len(ch):
        worst = ch["median_px"].max()
        txt = f"{len(ch)} layout changes; mismatch after the frame fit: median up to {fmt(worst)} px"
        after = None
        if el is not None and len(el):
            after = el["median_after_px"].max()
            txt += f"; {len(el)} corrected with a displacement field, leaving up to {fmt(after)} px (median)"
        left = max([v for v in [after if after is not None else None] if v is not None] +
                   ch.loc[~ch["z_after"].isin(el["z_after"] if el is not None and len(el) else []), "median_px"].tolist() + [0.0])
        out.append(("Layout changes", esc(txt), "remaining median mismatch <= 5 px (align.elastic_min_px)", PASS if left <= 5 else ATTN, ""))
    # tile seam brightness steps
    ss = read_csv(qc / "seam_steps.csv")
    if ss is None:
        out.append(("Tile seam brightness steps", "", "", NOTRUN, cmd("python tools/seam_steps.py CONFIG OLD_VOLUME Z,Z,... NAME=DIR") + " (see the tool's docstring)"))
    else:
        cols = [c for c in ss.columns if c not in ("z", "seam", "class", "stretches")]
        txt = "; ".join(f"{c}: median |step| {fmt(ss[c].abs().median())} greys, largest {fmt(ss[c].abs().max())}" for c in cols)
        out.append(("Tile seam brightness steps", esc(txt), "information", INFO, f'<a href="{esc(rel(qc / "seam_steps.csv", qc))}">seam_steps.csv</a>'))
    # contrast standard
    i = cfg.get("intensity") or {}
    name = f"intensity_reference_{i['output']}" if i.get("output") else "intensity_reference"
    ref = read_json(qc / f"{name}.json") if R.get("own_config") else None
    if i.get("window") != "reference":
        out.append(("Contrast standard", esc(f"window: {i.get('window')} (no reference standard set)"), "", INFO, ""))
    elif ref is None:
        out.append(("Contrast standard", "", "", NOTRUN, cmd("./run_pipeline.sh CONFIG --steps refmask,refstats,intensity")))
    else:
        p, tgt = ref["predicted_new"], ref["reference"]
        m, c = p["median_grey_p5_p50_p95"], p["contrast_p5_p50_p95"]
        txt = (f"neuropil median {m[1]:.1f} (5th-95th percentile of {ref.get('samples')} sampled slices: {m[0]:.1f}-{m[2]:.1f}; target {tgt['median']}); "
               f"p5-p95 width {c[1]:.1f} ({c[0]:.1f}-{c[2]:.1f}; target {tgt['contrast']}); within the band: median {100 * p['in_band_median']:.1f} %, "
               f"width {100 * p['in_band_contrast']:.1f} % of samples; tissue clipped at 0 or 255: median {100 * p['clipped_median']:.2f} %, "
               f"largest {100 * p['clipped_max']:.2f} %; clipping warnings at {len(p.get('clip_warn_slices', []))} sampled slices")
        ok = (min(p["in_band_median"], p["in_band_contrast"]) >= LIMITS["reference_in_band"] and p["clipped_max"] <= LIMITS["clipped_max"]
              and not p.get("clip_warn_slices"))
        notes = "".join(f"<br>{esc(f)}" for f in ref.get("flags", [])[:8])
        out.append(("Contrast standard (predicted from the sampled slices)", esc(txt),
                    f"in band >= {100 * LIMITS['reference_in_band']:g} % of samples, clipped <= {100 * LIMITS['clipped_max']:g} %",
                    PASS if ok else ATTN, f'<a href="{esc(name)}.json">{esc(name)}.json</a>{notes}'))
    clip = read_csv(qc / "clipping" / "clipping.csv")
    if clip is None:
        out.append(("Clipping measured in the volume", "", "", NOTRUN, cmd("python tools/clipping_check.py CONFIG Z,Z,...")))
    else:
        num = clip.select_dtypes("number").drop(columns=[c for c in ("z",) if c in clip], errors="ignore")
        txt = "; ".join(f"{c}: max {fmt(num[c].max())}" for c in num.columns[:6])
        out.append(("Clipping measured in the volume", esc(txt), "information", INFO, '<a href="clipping/clipping.csv">clipping/clipping.csv</a>'))
    # flicker
    fl_dirs = [Path(R["flicker"])] if R.get("flicker") else sorted(p.parent for p in qc.glob("zflicker*/zflicker.json"))
    fl_rows = []
    for d in fl_dirs:
        j = read_json(Path(d) / "zflicker.json")
        for sname, v in ((j or {}).get("series") or {}).items():
            fl_rows.append((Path(d).name, sname, v.get("flicker_common_sd"), v.get("flicker_own_sd"), v.get("planes")))
    if not fl_rows:
        out.append(("Flicker along z", "", "", NOTRUN,
                    cmd("sbatch slurm/zflicker.sbatch --out OUTPUT_DIR/qc/zflicker --config CONFIG NAME=VOLUME.ome.zarr")))
    else:
        vol_rows = [r for r in fl_rows if r[1] not in ("raw (fixed window)", "windows (correction)")]
        worst = max((r[2] for r in vol_rows if r[2] is not None), default=np.nan)
        det = table(["folder", "series", "slice-wide sd (greys)", "each region's own sd", "planes"],
                    [[esc(a), esc(b), fmt(c), fmt(d_), fmt(e)] for a, b, c, d_, e in fl_rows])
        out.append(("Flicker along z", esc(f"{len(fl_rows)} series measured; largest slice-wide sd among the volumes and the prediction "
                                           f"('raw - windows'): {fmt(worst)} greys; the raw tiles and the windows are listed for comparison"),
                    f"slice-wide sd <= {LIMITS['flicker_common_sd']:g} grey", PASS if worst <= LIMITS["flicker_common_sd"] else ATTN, det))
    # slice thickness
    pos = read_csv(step_path(cfg, "zcorrect", "positions.csv"))
    if pos is None:
        out.append(("Slice thickness", "", "", NOTRUN, cmd("./run_pipeline.sh CONFIG --steps zcorrect") + " (with zcorrect.enabled: true)"))
    else:
        sp = np.diff(pos["position_nm"].to_numpy(float)) / np.maximum(np.diff(pos["z"].to_numpy(float)), 1)
        nominal = float(np.median(sp)) if len(sp) else np.nan
        cv = float(1.4826 * np.median(np.abs(sp - nominal)) / nominal) if len(sp) and nominal else np.nan
        txt = (f"estimated spacing: median {fmt(nominal)} nm, 5th-95th percentile {fmt(np.percentile(sp, 5))}-{fmt(np.percentile(sp, 95))} nm, "
               f"robust scatter {100 * cv:.1f} % of the median; " + ("applied in this volume" if (R.get("render") or {}).get("zcorrected") else "not applied in this volume"))
        out.append(("Slice thickness", esc(txt), f"scatter <= {100 * LIMITS['thickness_cv']:g} %", PASS if cv <= LIMITS["thickness_cv"] else ATTN, ""))
    # curtains and slice quality
    sq = read_json(qc / "slice_quality" / "summary.json")
    sq_cmd = ("A=$(sbatch --parsable --array=0-19 slurm/slice_quality.sbatch CONFIG); "
              "sbatch --dependency=afterok:$A slurm/slice_quality.sbatch CONFIG merge")
    if sq is None:
        out.append(("Curtains (raw tiles)", "", "", NOTRUN, cmd(sq_cmd)))
        out.append(("Slice quality flags", "", "", NOTRUN, cmd(sq_cmd)))
    else:
        c = sq["counts"].get("curtain", {"slices": 0, "major": 0, "minor": 0})
        ds = (R.get("render") or {}).get("destreak") or cfg.get("destreak") or {}
        out.append(("Curtains (measured on the raw tiles, before any curtain removal)",
                    esc(f"{c['slices']} of {sq['slices']} slices have a tile with curtains across more than "
                        f"{100 * sq['limits']['curtain']['limit']:g} % of its measured windows; curtain removal in this volume: "
                        + ("on" if ds.get("enabled") else "off")), "information", INFO, ""))
        n = sq["slices"]
        flags = read_csv(qc / "slice_flags.csv", dtype={"tile": str})
        major = flags[(flags["severity"] == "major") & ~flags["reason"].isin(["curtain", "excluded"])]["z"].nunique() if flags is not None else 0
        txt = f"{sq['flagged_slices']} of {n} slices carry a flag; {major} with a major flag other than curtains"
        out.append(("Slice quality flags", esc(txt), f"slices with a major flag (curtains aside) <= {100 * LIMITS['major_flag_share']:g} %",
                    PASS if major <= LIMITS["major_flag_share"] * n else ATTN, "details below"))
    return out


def checks_section(rows):
    body = [[esc(a), b, esc(c), mark(s), d] for a, b, c, s, d in rows]
    return table(["check", "numbers", "limit", "", "more"], body, "checks")


def quality_section(R):
    qc = R["qc"]
    sq = read_json(qc / "slice_quality" / "summary.json")
    flags = read_csv(qc / "slice_flags.csv", dtype={"tile": str})
    if sq is None or flags is None:
        return ("<p>" + mark(NOTRUN) + " <code>A=$(sbatch --parsable --array=0-19 slurm/slice_quality.sbatch CONFIG); "
                "sbatch --dependency=afterok:$A slurm/slice_quality.sbatch CONFIG merge</code></p>")
    out = ("<p>From <code>tools/slice_quality.py</code>, measured on the raw tiles and the preview thumbnails. Flags mark slices to look "
           "at; no slice is excluded or changed because of them. A flag compares a tile with its own trend along z or with a fixed limit "
           "(the tool's docstring gives each rule).</p>")
    rows = []
    for reason, c in sq["counts"].items():
        lim = sq["limits"].get(reason, {})
        rows.append([esc(reason), fmt(c["slices"]), fmt(c["major"]), fmt(c["minor"]),
                     esc(f"{lim.get('metric')} {lim.get('direction')} {lim.get('limit')}") if lim else ""])
    out += table(["reason", "slices", "major flags", "minor flags", "rule"], rows) if rows else "<p>No slice is flagged.</p>"
    f = flags[~flags["reason"].isin(["excluded"])].copy()
    if len(f):
        f["rank"] = (f["severity"] == "major").astype(int)
        worst = f.sort_values(["rank", "zscore"], ascending=[False, False], key=lambda s: s.abs() if s.name == "zscore" else s)
        worst = worst.groupby("reason", sort=True).head(5)
        out += "<h3>Worst flagged slices per reason</h3>" + table(
            ["reason", "z", "tile", "value", "limit", "severity"],
            [[esc(r.reason), r.z, esc(r.tile), fmt(r.value), fmt(r.limit), esc(r.severity)] for r in worst.sort_values(["reason", "z"]).itertuples()])
    links = ['<a href="slice_flags.csv">slice_flags.csv</a>', '<a href="slice_quality.csv">slice_quality.csv</a>']
    links += [f'<a href="slice_quality/{esc(s)}">{esc(s)}</a>' for s in sq.get("sheets", [])]
    return out + "<p>Tables and contact sheets (flagged slice next to its neighbours): " + ", ".join(links) + "</p>"


def embed(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    if img.shape[1] > FIGURE_WIDTH:
        s = FIGURE_WIDTH / img.shape[1]
        img = cv2.resize(img, (FIGURE_WIDTH, max(1, round(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    return base64.b64encode(buf.tobytes()).decode() if ok else None


def figures_section(cfg, R):
    qc = R["qc"]
    i = (cfg.get("intensity") or {}).get("output") if R.get("own_config") else None
    rs = (cfg.get("refstats") or {}).get("output") if R.get("own_config") else None
    fl = Path(R["flicker"]) if R.get("flicker") else next(iter(sorted(p.parent for p in qc.glob("zflicker*/zflicker.png"))), None)
    wanted = [("stitch.png", "Stitch: tile offsets over z"), ("align_drift.png", "Align: drift and residuals over z"),
              ("tile_jumps.png", "Tile jumps and alignment residuals over time"),
              (f"intensity_{i}.png" if i else "intensity.png", "Display levels over z"),
              (f"intensity_reference_{i}.png" if i else "intensity_reference.png", "Contrast standard: predicted neuropil grey and width"),
              (f"refstats_{rs}.png" if rs else "refstats.png", "Neuropil statistics at the sampled slices"),
              ("zcorrect.png", "Slice thickness"), ("slice_quality.png", "Slice quality metrics over z")]
    if fl is not None:
        wanted.insert(6, (rel(Path(fl) / "zflicker.png", qc), "Flicker along z"))
    out = []
    for name, caption in wanted:
        b64 = embed(qc / name)
        if b64:
            out.append(f'<figure><img alt="{esc(caption)}" src="data:image/png;base64,{b64}"><figcaption>{esc(caption)} '
                       f'(<a href="{esc(name)}">{esc(name)}</a>)</figcaption></figure>')
    return "".join(out) or "<p>No qc plots found.</p>"


def dir_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.lstat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total


def files_section(cfg, R, disk):
    qc, out_dir = R["qc"], Path(cfg["output_dir"])
    rows = []
    vol = out_dir / R["volume"]
    r = R.get("render")
    if r:
        raw_gb = float(np.prod([float(v) for v in r["shape"]])) / 1e9
        size = f"{raw_gb:,.1f} GB uncompressed at full resolution (8 bit)"
        if disk and vol.exists():
            size += f"; {dir_size(vol) / 1e9:,.1f} GB on disk, all scales"
        done = len(list((step_path(cfg, "render", R["stem"], "done")).glob("slab_*")))
        slabs = -(-int(r["shape"][0]) // int(r["settings"].get("slab", 64)))
        rows.append([esc(rel(vol, qc)), "OME-Zarr volume" + ("" if vol.exists() else " (not found)"), esc(size),
                     esc(f"rendered slabs {done} of {slabs}")])
    else:
        rows.append([esc(rel(vol, qc)), "OME-Zarr volume", mark(NOTRUN), "<code>./run_pipeline.sh CONFIG --steps render,pyramid</code>"])
    exp = step_path(cfg, "precomputed", R["stem"] + ".precomputed", "export.json")
    if exp.exists():
        rows.append([esc(exp.parent.name), "Neuroglancer precomputed export", "", esc("export plan written (its destination is in work/precomputed/, not shown here)")])
    for p in sorted(out_dir.glob("*.tif")):
        if p.name.startswith(R["stem"]):
            rows.append([esc(rel(p, qc)), "quick-look TIFF", f"{p.stat().st_size / 1e9:,.1f} GB", ""])
    for p in sorted(qc.rglob("view_*.txt"))[:30]:
        rows.append([f'<a href="{esc(rel(p, qc))}">{esc(rel(p, qc))}</a>', "neuroglancer link file", "", ""])
    for name, what in (("check_report.md", "raw data check report"), ("slice_flags.csv", "slice quality flags"), ("slice_quality.csv", "slice quality metrics")):
        if (qc / name).exists():
            rows.append([f'<a href="{esc(name)}">{esc(name)}</a>', what, "", ""])
    return table(["file (relative to this page)", "what", "size", "note"], rows)


CSS = """
:root{--bg:#f4f6f7;--surface:#fff;--ink:#15191d;--ink-2:#434b55;--muted:#6a737e;--rule:#d6dbe0;--accent:#0c6a73;
--ok:#1d6b3a;--ok-soft:#dcefe2;--warn:#8a5a00;--warn-soft:#f6ead2;--none-soft:#e6e9ec}
@media (prefers-color-scheme: dark){:root{--bg:#101316;--surface:#171b1f;--ink:#e7eaed;--ink-2:#b9c1c9;--muted:#8a939d;--rule:#2a3036;
--accent:#62c3ca;--ok:#7fd09a;--ok-soft:#173224;--warn:#e0b45c;--warn-soft:#3a2e14;--none-soft:#252b31;color-scheme:dark}}
body{background:var(--bg);color:var(--ink);font-family:system-ui,-apple-system,"Segoe UI",sans-serif;font-size:15.5px;line-height:1.55;
margin:0;padding:32px 16px 56px}
.page{max-width:68rem;margin:0 auto;display:grid;gap:36px}
h1{font-size:1.9rem;margin:0 0 6px;line-height:1.2} h2{font-size:1.3rem;margin:0 0 12px} h3{font-size:1.02rem;margin:18px 0 8px}
p{margin:0 0 10px;color:var(--ink-2)} a{color:var(--accent)} .sub{color:var(--muted);font-size:.93rem}
section{background:var(--surface);border:1px solid var(--rule);border-radius:8px;padding:20px 22px;min-width:0}
.scroll{overflow-x:auto} table{border-collapse:collapse;width:100%;font-size:.92rem}
th,td{text-align:left;vertical-align:top;padding:6px 10px;border-bottom:1px solid var(--rule)}
th{color:var(--muted);font-weight:600;font-size:.82rem} td{font-variant-numeric:tabular-nums}
table.facts td:first-child{color:var(--ink-2);width:34%} table.facts thead{display:none}
table.checks td:first-child{font-weight:600;min-width:9rem} table.checks td:nth-child(3){color:var(--muted)}
code{font-family:ui-monospace,"SF Mono",Menlo,monospace;font-size:.86em;background:var(--none-soft);padding:1px 4px;border-radius:3px;overflow-wrap:anywhere}
.mark{display:inline-block;padding:1px 9px;border-radius:10px;font-size:.82rem;font-weight:600;white-space:nowrap}
.mark.pass{background:var(--ok-soft);color:var(--ok)} .mark.attention{background:var(--warn-soft);color:var(--warn)}
.mark.not-run,.mark.info{background:var(--none-soft);color:var(--muted)}
figure{margin:0 0 22px} figure img{max-width:100%;height:auto;border:1px solid var(--rule);background:#fff}
figcaption{color:var(--muted);font-size:.88rem;margin-top:4px}
"""


def build(cfg, R, disk=False):
    data_html, facts = dataset_section(cfg, R)
    rows = checks(cfg, R, facts)
    counts = {s: sum(1 for r in rows if r[3] == s) for s in (PASS, ATTN, NOTRUN)}
    title = f"QC summary: {Path(R['volume']).name}"
    head = (f"<h1>{esc(title)}</h1><p class=\"sub\">Dataset {esc(cfg.get('name') or '')}. Written {datetime.now():%Y-%m-%d %H:%M} by "
            f"<code>tools/qc_summary.py</code> from the run's existing outputs. Checks: {counts[PASS]} pass, {counts[ATTN]} attention, "
            f"{counts[NOTRUN]} not run. The limits are working values stated in the table, not validated standards: "
            "&ldquo;attention&rdquo; means the numbers are worth looking at.</p>")
    parts = [("Dataset", data_html), ("Processing settings", settings_section(cfg, R)), ("Checks", checks_section(rows)),
             ("Slice quality flags", quality_section(R)), ("Figures", figures_section(cfg, R)), ("Files", files_section(cfg, R, disk))]
    body = "".join(f"<section><h2>{esc(t)}</h2>{h}</section>" for t, h in parts)
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>{esc(title)}</title><style>{CSS}</style></head><body><div class=\"page\"><header>{head}</header>{body}</div></body></html>")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("config")
    p.add_argument("--volume", help="volume folder name in output_dir (default: the config's render.name)")
    p.add_argument("--out", help="HTML path (default <output_dir>/qc/summary_<volume>.html)")
    p.add_argument("--flicker", help="the zflicker output folder to report")
    p.add_argument("--disk-size", action="store_true")
    p.add_argument("--raw-dir")
    p.add_argument("--output-dir")
    a = p.parse_args(argv)
    cfg = load_config(a.config, a.raw_dir, a.output_dir, step_defaults())
    own = render.volume_name(cfg)
    volume = a.volume or own
    stem = Path(volume).name.removesuffix(".zarr").removesuffix(".ome")
    qc = Path(cfg["output_dir"]) / "qc"
    R = {"qc": qc, "volume": volume, "stem": stem, "own_config": Path(volume).name == Path(own).name, "commit": commit(),
         "render": read_json(step_path(cfg, "render", stem, "render.json")), "flicker": a.flicker,
         "plan": read_csv(step_path(cfg, "render", stem, "tiles.csv"), usecols=["z"])}
    out = Path(a.out) if a.out else qc / f"summary_{stem}.html"
    page = build(cfg, R, a.disk_size)
    atomic_write(out, lambda t: Path(t).write_text(page, encoding="utf-8"))
    print(f"qc_summary: wrote {out} ({len(page) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
