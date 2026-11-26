"""A one-page QC summary of a rendered volume: one self-contained HTML file.

    python tools/qc_summary.py CONFIG [--volume NAME] [--out HTML] [--flicker DIR[:SERIES,...] ...]
                               [--predict-flicker] [--chain FILE] [--disk-size]

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

--flicker DIR[:SERIES,...]   a tools/zflicker.py output folder holding this volume's flicker, and the series
                in it that are this volume (as named when zflicker ran; default: all of them, without the
                raw tiles and the windows); may be given several times, e.g. the prediction and the
                measurement. A folder name alone is looked for in the qc folder. Without --flicker every
                qc/zflicker*/ folder is listed, without a pass / attention mark (the page cannot tell
                which series is this volume).
--predict-flicker   first write the flicker the levels predict (tools/zflicker.py --config, no volume read) to
                qc/zflicker_levels[_<intensity.output>]/ and mark its "raw - windows" series as this volume's
--chain FILE    the job list run_pipeline.sh wrote for this chain (name, Slurm job id per line): the page then
                starts with the run's outcome: every job's state from Slurm's accounting (sacct), and when a job
                failed, which one, the last lines of its log, and that the jobs after it did not start
--disk-size     also add up the volume's size on disk (walks every file: slow on large volumes)

The page starts with one verdict: STOPPED (a job of the chain failed), ATTENTION (checks to look at, listed) or
PASSED (every check that ran passed), and ends with the decisions that stay with a person.

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
from pipeline.config import REMOVED_STEPS, load_config, step_defaults, step_path  # noqa: E402

LIMITS = {
    "stitch_failed_samples": 0,        # stitched samples without a solution
    "stitch_residual_px": 5.0,         # median residual of the tile pairs used (the pipeline's own stitch.refine_px scale)
    "align_rejected_share": 0.005,     # share of slice pairs the alignment rejected
    "align_rms_px": 5.0,               # median residual of the accepted slice pairs
    "big_jump_px": 15.0,               # a tile movement between two slices counted as a large jump
    "reference_in_band": 0.95,         # share of sampled slices with the neuropil within the standard's band
    "clipped_max": 0.01,               # largest predicted share of tissue clipped at 0 or 255 in a sampled slice
    "flicker_common_sd": 1.0,          # slice-wide flicker of the volume, greys (robust sd)
    "major_flag_share": 0.01,          # share of slices with a major slice-quality flag
}
PASS, ATTN, NOTRUN, INFO = "pass", "attention", "not run", "info"
# Shown only where an earlier run left the step's results (work/zcorrect/positions.csv, qc/zcorrect.png) or a
# volume's render.json says the correction was applied.
ZCORRECT_REMOVED = ("the slice-thickness step (zcorrect) is no longer part of the pipeline "
                    f"({REMOVED_STEPS['zcorrect'][1]})")
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
            ("Render", f"downsample {rs.get('downsample')}, blend {rs.get('blend')}, rotation {rs.get('rotate_deg', 0)} deg, "
                       f"{rs.get('num_scales')} scales, compression {rs.get('compression')}"),
            ("CLAHE", (f"on (clip {rs['clahe'].get('clip_limit')}, {rs['clahe'].get('tile_px')} px)" if (rs.get("clahe") or {}).get("enabled") else "off")),
            ("em-pipe commit that wrote this page", R["commit"])]
    if r.get("zcorrected"):   # only a volume made while the step existed: render no longer applies it
        rows.insert(-3, ("Slice-thickness correction (zcorrect)", f"applied when this volume was rendered; {ZCORRECT_REMOVED}"))
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
    chosen = R.get("flicker") or []
    fl_dirs = [(d, names) for d, names in chosen] or [(p.parent, None) for p in sorted(qc.glob("zflicker*/zflicker.json"))]
    fl_rows = []
    for d, names in fl_dirs:
        j = read_json(Path(d) / "zflicker.json")
        for sname, v in ((j or {}).get("series") or {}).items():
            mine = bool(chosen) and (sname in names if names else sname not in ("raw (fixed window)", "windows (correction)"))
            fl_rows.append((Path(d).name, sname, v.get("flicker_common_sd"), v.get("flicker_own_sd"), v.get("planes"), mine))
    if not fl_rows:
        out.append(("Flicker along z", "", "", NOTRUN,
                    cmd("sbatch slurm/zflicker.sbatch --out OUTPUT_DIR/qc/zflicker --config CONFIG NAME=VOLUME.ome.zarr")))
    else:
        det = table(["folder", "series", "slice-wide sd (greys)", "each region's own sd", "planes", "this volume"],
                    [[esc(a), esc(b), fmt(c), fmt(d_), fmt(e), "yes" if m else ""] for a, b, c, d_, e, m in fl_rows])
        own = [r for r in fl_rows if r[5] and r[2] is not None]
        if own:
            worst = max(r[2] for r in own)
            txt = ("slice-wide flicker (robust sd of the brightness the whole plane shares, against a running mean along z): "
                   + "; ".join(f"{r[1]} ({r[0]}): {fmt(r[2])} greys" for r in own)
                   + ". 'raw - windows' is the prediction from the raw tiles and the display windows, without a render")
            out.append(("Flicker along z", esc(txt), f"slice-wide sd <= {LIMITS['flicker_common_sd']:g} grey",
                        PASS if worst <= LIMITS["flicker_common_sd"] else ATTN, det))
        else:
            out.append(("Flicker along z", esc(f"{len(fl_rows)} series in {len(fl_dirs)} folders; which of them is this volume was not "
                                               "given (--flicker), so they are listed without a mark"),
                        f"slice-wide sd <= {LIMITS['flicker_common_sd']:g} grey", INFO, det))
    # slice thickness: only what an earlier run of the former zcorrect step left; no row otherwise
    pos = read_csv(step_path(cfg, "zcorrect", "positions.csv"))
    if pos is not None and len(pos) > 1:
        sp = np.diff(pos["position_nm"].to_numpy(float)) / np.maximum(np.diff(pos["z"].to_numpy(float)), 1)
        nominal = float(np.median(sp))
        cv = float(1.4826 * np.median(np.abs(sp - nominal)) / nominal) if nominal else np.nan
        txt = (f"estimated spacing: median {fmt(nominal)} nm, 5th-95th percentile {fmt(np.percentile(sp, 5))}-{fmt(np.percentile(sp, 95))} nm, "
               f"robust scatter {100 * cv:.1f} % of the median; " + ("applied in this volume" if (R.get("render") or {}).get("zcorrected") else "not applied in this volume"))
        out.append(("Slice thickness (from an earlier run)", esc(txt), "no limit set: listed for information", INFO, esc(ZCORRECT_REMOVED)))
    # curtains and slice quality
    sq = read_json(qc / "slice_quality" / "summary.json")
    sq_cmd = ("A=$(sbatch --parsable --array=0-19 slurm/slice_quality.sbatch CONFIG); "
              "sbatch --dependency=afterok:$A slurm/slice_quality.sbatch CONFIG merge")
    if sq is None:
        out.append(("Curtains (raw tiles)", "", "", NOTRUN, cmd(sq_cmd)))
        out.append(("Slice quality flags", "", "", NOTRUN, cmd(sq_cmd)))
    else:
        n = sq["slices"]
        flags = read_csv(qc / "slice_flags.csv", dtype={"tile": str})
        flags = flags if flags is not None else pd.DataFrame(columns=["z", "reason", "severity"])
        cur = flags[flags["reason"] == "curtain"]
        ds = (R.get("render") or {}).get("destreak") or cfg.get("destreak") or {}
        out.append(("Curtains (measured on the raw tiles, before any curtain removal)",
                    esc(f"{cur['z'].nunique()} of {n} slices have a tile with at least one curtained window (stripe amplitude along the "
                        f"curtain direction over {sq['settings'].get('curtain_ratio')} times that across it); in "
                        f"{cur.loc[cur['severity'] == 'major', 'z'].nunique()} slices a tenth or more of a tile's windows are curtained. "
                        "Curtain removal in this volume: " + ("on" if ds.get("enabled") else "off")), "information", INFO, ""))
        major = flags[(flags["severity"] == "major") & ~flags["reason"].isin(["curtain", "excluded"])]["z"].nunique()
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
        # how far beyond its limit: below-limits count downwards
        below = {r for r, lim in sq["limits"].items() if lim.get("direction") == "below"}
        f["beyond"] = np.where(f["reason"].isin(below), -f["value"], f["value"].abs())
        worst = f.sort_values("beyond", ascending=False).groupby("reason", sort=True).head(5)
        out += "<h3>Worst flagged slices per reason</h3>" + table(
            ["reason", "z", "tile", "value", "limit", "severity"],
            [[esc(r.reason), r.z, esc(r.tile), fmt(r.value), fmt(r.limit), esc(r.severity)] for r in worst.sort_values(["reason", "beyond"], ascending=[True, False]).itertuples()])
    links = ['<a href="slice_flags.csv">slice_flags.csv</a>', '<a href="slice_quality.csv">slice_quality.csv</a>']
    links += [f'<a href="slice_quality/{esc(s)}">{esc(s)}</a>' for s in sq.get("sheets", [])]
    return out + "<p>Tables and contact sheets (flagged slice next to its neighbours): " + ", ".join(links) + "</p>"


def embed(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR) if Path(path).is_file() else None
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
    fl = R["flicker"][0][0] if R.get("flicker") else next(iter(sorted(p.parent for p in qc.glob("zflicker*/zflicker.png"))), None)
    wanted = [("stitch.png", "Stitch: tile offsets over z"), ("align_drift.png", "Align: drift and residuals over z"),
              ("tile_jumps.png", "Tile jumps and alignment residuals over time"),
              (f"intensity_{i}.png" if i else "intensity.png", "Display levels over z"),
              (f"intensity_reference_{i}.png" if i else "intensity_reference.png", "Contrast standard: predicted neuropil grey and width"),
              (f"refstats_{rs}.png" if rs else "refstats.png", "Neuropil statistics at the sampled slices"),
              ("zcorrect.png", "Slice thickness, from an earlier run: " + ZCORRECT_REMOVED),
              ("slice_quality.png", "Slice quality metrics over z")]
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
    for p in sorted(qc.rglob("view_*.txt"))[:200]:
        try:   # only the link files that show this volume
            if Path(R["volume"]).name not in p.read_text(errors="ignore"):
                continue
        except OSError:
            continue
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
    run_html, stopped = run_section(cfg, R)
    counts = {s: sum(1 for r in rows if r[3] == s) for s in (PASS, ATTN, NOTRUN)}
    title = f"QC summary: {Path(R['volume']).name}"
    cls, word, detail = verdict(rows, stopped)
    head = (f"<h1>{esc(title)}</h1><p class=\"verdict {cls}\"><b>{word}.</b> {detail}</p>"
            f"<p class=\"sub\">Dataset {esc(cfg.get('name') or '')}. Written {datetime.now():%Y-%m-%d %H:%M} by "
            f"<code>tools/qc_summary.py</code> from the run's existing outputs. Checks: {counts[PASS]} pass, {counts[ATTN]} attention, "
            f"{counts[NOTRUN]} not run. The limits are working values stated in the table, not validated standards: "
            "&ldquo;attention&rdquo; means the numbers are worth looking at.</p>")
    parts = [("Dataset", data_html + before_run_section(R)), ("Processing settings", settings_section(cfg, R)),
             ("Checks", checks_section(rows)), ("Slice quality flags", quality_section(R)), ("Figures", figures_section(cfg, R)),
             ("Files", files_section(cfg, R, disk)), ("The run", run_html), ("Decisions for a person", decisions_section(rows))]
    body = "".join(f"<section><h2>{esc(t)}</h2>{h}</section>" for t, h in parts)
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>{esc(title)}</title><style>{CSS}{VERDICT_CSS}</style></head><body><div class=\"page\"><header>{head}</header>{body}</div></body></html>")


VERDICT_CSS = (".verdict{padding:10px 14px;border-radius:6px;border:2px solid var(--rule);margin:10px 0;color:var(--ink)}"
               ".verdict.pass{background:var(--ok-soft);border-color:var(--ok)}"
               ".verdict.attention{background:var(--warn-soft);border-color:var(--warn)}"
               ".verdict.stopped{background:var(--warn-soft);border-color:#b3261e}"
               "pre{white-space:pre-wrap;font-size:12px;background:var(--none-soft);padding:8px;border-radius:4px}")


BAD_STATES = ("FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL", "DEADLINE", "PREEMPTED")
DECISIONS = (
    ("Leaving out files or slices", "Nothing is excluded automatically. The raw data check and the slice quality flags above say "
     "what to look at; exclusions are stated in the config (check.known_issues, check.exclude_slices)."),
    ("Where the contrast standard is measured", "By default in all reference tissue of every sampled slice. If part of the volume "
     "should not count (tissue other volumes do not have), set refstats.region and run refstats and intensity again."),
    ("The reference-tissue mask for this stain", "The mask classifier was trained on our stain. Look at refmask.png (Figures): if it "
     "marks the wrong tissue, new labels and a retrained model are needed (tools/refmask_train.py)."),
    ("Judging the result", "This page reports numbers against working limits; whether the volume is good is judged by looking at "
     "it (the view link under Files) and at the figures."),
    ("Uploading to a bucket", "Not part of this run: an upload is a config of its own with its destination."),
    ("Deleting volumes or work folders", "Nothing is deleted by the pipeline; superseded volumes stay until a person removes them."),
)


def read_chain(path):
    """[(job name, job id)] from the file run_pipeline.sh wrote; None when there is none."""
    path = Path(path)
    if not path.is_file():
        return None
    rows = [line.split("\t") for line in path.read_text().splitlines() if line.strip()]
    return [(r[0], r[1]) for r in rows if len(r) >= 2]


def sacct_text(ids):
    """Slurm's accounting lines for the jobs (one per job or array task), '' when sacct cannot be asked."""
    try:
        r = subprocess.run(["sacct", "-n", "-P", "-X", "-j", ",".join(ids), "--format=JobID,State,ExitCode,ElapsedRaw,CPUTimeRAW"],
                           capture_output=True, text=True, timeout=120)
        return r.stdout if r.returncode == 0 else ""
    except Exception:
        return ""


def job_states(chain, text):
    """Per job of the chain: {name, id, state, tasks, elapsed_s (longest task), cpu_s (all tasks)} from sacct's
    lines. An array's state is its worst task's; a job sacct does not list is 'unknown'."""
    per = {}
    for line in text.splitlines():
        f = line.split("|")
        if len(f) < 5:
            continue
        base = f[0].split("_")[0].split(".")[0]
        d = per.setdefault(base, {"states": [], "elapsed": 0, "cpu": 0})
        d["states"].append(f[1].split()[0] if f[1].strip() else "unknown")
        d["elapsed"] = max(d["elapsed"], int(f[3] or 0))
        d["cpu"] += int(f[4] or 0)
    order = (*BAD_STATES, "CANCELLED", "RUNNING", "PENDING", "COMPLETED")
    out = []
    for name, jid in chain:
        d = per.get(jid)
        if d is None:
            out.append({"name": name, "id": jid, "state": "unknown", "tasks": 0, "elapsed_s": 0, "cpu_s": 0})
            continue
        state = min(d["states"], key=lambda x: order.index(x) if x in order else len(order) - 1.5)
        out.append({"name": name, "id": jid, "state": state, "tasks": len(d["states"]), "elapsed_s": d["elapsed"], "cpu_s": d["cpu"]})
    return out


def log_tail(logs, name, jid, lines=4):
    """The last ERROR lines (else the last lines) of a job's log files in work/logs, as text."""
    found = sorted(Path(logs).glob(f"{name}-{jid}*.out")) if Path(logs).is_dir() else []
    out = []
    for f in found:
        try:
            text = f.read_text(errors="replace").splitlines()
        except OSError:
            continue
        err = [t for t in text if " ERROR " in t or "Error" in t or "error:" in t.lower()]
        keep = (err or [t for t in text if t.strip()])[-lines:]
        if err or len(found) == 1:
            out += [f"{f.name}: {t.strip()[:600]}" for t in keep]
        if len(out) >= 3 * lines:
            break
    return out


def run_section(cfg, R):
    """(html, stopped: None or a sentence) for the chain this page ends."""
    chain = R.get("chain")
    if chain is None:
        return ("<p>This page was written outside a job chain (no <code>--chain</code>): the jobs' states are not listed.</p>", None)
    if not chain:
        return ("<p>The chain held no job before this page.</p>", None)
    jobs = job_states(chain, R.get("sacct") if R.get("sacct") is not None else sacct_text([j for _, j in chain]))
    bad = next((j for j in jobs if j["state"] in BAD_STATES), None)
    stopped = None
    notes = ""
    if bad is not None:
        after = [j["name"] for j in jobs[jobs.index(bad) + 1:] if j["state"] != "COMPLETED"]
        tail = log_tail(Path(cfg["output_dir"]) / "work" / "logs", bad["name"], bad["id"])
        stopped = (f"The chain stopped at {bad['name']} (job {bad['id']}, {bad['state']})."
                   + (f" {len(after)} later jobs did not run: {', '.join(after[:12])}{' ...' if len(after) > 12 else ''}." if after else ""))
        notes = ("<p><b>" + esc(stopped) + "</b></p>"
                 + ("<p>From its log:</p><pre>" + esc("\n".join(tail)) + "</pre>" if tail else
                    f"<p>Its log: <code>work/logs/{esc(bad['name'])}-{esc(bad['id'])}*.out</code></p>"))
    elif any(j["state"] == "unknown" for j in jobs):
        notes = "<p>Slurm's accounting gave no state for some jobs (sacct not available here, or not yet written).</p>"
    hours = sum(j["cpu_s"] for j in jobs) / 3600.0
    rows = [[esc(j["name"]), esc(j["id"]), esc(j["state"]), fmt(j["tasks"]) if j["tasks"] else "",
             f"{j['elapsed_s'] / 60:.1f}" if j["tasks"] else "", f"{j['cpu_s'] / 3600:.2f}" if j["tasks"] else ""] for j in jobs]
    return (notes + f"<p>{len(jobs)} jobs before this page; {hours:,.1f} core-hours in all (allocated cores x run time).</p>"
            + table(["job", "Slurm id", "state", "tasks", "longest task (min)", "core-hours"], rows), stopped)


def before_run_section(R):
    """What the em-pipe command found before it submitted the chain (qc/em_pipe_detected.json), as html; ''
    when the run was not started by it."""
    d = read_json(R["qc"] / "em_pipe_detected.json")
    if not d:
        return ""
    rows = [["Raw files", esc(d.get("raw", ""))], ["EsB files", esc(d.get("esb", ""))], ["Curtain direction", esc(d.get("direction", ""))]]
    return "<h3>Found before the run (em-pipe)</h3>" + table(["", ""], [r for r in rows if r[1]])


def verdict(rows, stopped):
    """(css class, headline, detail html) of the page's one-line outcome."""
    attn = [r[0] for r in rows if r[3] == ATTN]
    notrun = [r[0] for r in rows if r[3] == NOTRUN]
    npass = sum(1 for r in rows if r[3] == PASS)
    if stopped:
        return "stopped", "STOPPED", esc(stopped) + " See &ldquo;The run&rdquo; below for its log; the checks listed are those of the steps that did run."
    if attn:
        return "attention", "ATTENTION", (f"{len(attn)} check{'s' if len(attn) != 1 else ''} to look at: " + esc("; ".join(attn))
                                          + f". {npass} passed" + (f", {len(notrun)} not run." if notrun else "."))
    return "pass", "PASSED", (f"Every check that ran passed ({npass})"
                              + (f"; not run: {esc('; '.join(notrun))}." if notrun else "."))


def decisions_section(rows):
    items = "".join(f"<li><b>{esc(t)}.</b> {esc(d)}</li>" for t, d in DECISIONS)
    attn = [r[0] for r in rows if r[3] == ATTN]
    lead = ("<p>Checks marked attention, to be looked at first: " + esc("; ".join(attn)) + ".</p>") if attn else ""
    return lead + "<p>The pipeline does not take these; each is stated in the config or done by a person.</p><ul>" + items + "</ul>"


def predict_flicker(cfg, config_path, qc):
    """Run tools/zflicker.py on the levels (no volume read); (folder, ['raw - windows']) or None when there are
    no levels yet."""
    from pipeline import intensity
    if not intensity.levels_path(cfg).exists() or not step_path(cfg, "preview", "stats.csv").exists():
        return None
    out = (cfg.get("intensity") or {}).get("output")
    folder = qc / ("zflicker_levels" + (f"_{out}" if out else ""))
    import importlib.util
    spec = importlib.util.spec_from_file_location("zflicker", ROOT / "tools" / "zflicker.py")
    zf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(zf)
    try:
        zf.main(["--out", str(folder), "--config", str(config_path)])
    except Exception as e:   # the page is still written
        print(f"qc_summary: the predicted flicker could not be measured ({type(e).__name__}: {e})")
        return None
    return folder, ["raw - windows"]


def flicker_arg(text, qc):
    """(folder, series names or None) from DIR[:SERIES,...]; a folder that does not exist as given is taken in qc."""
    d, _, names = text.partition(":")
    d = Path(d) if Path(d).is_dir() else qc / d
    return d, [n.strip() for n in names.split(",") if n.strip()] or None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("config")
    p.add_argument("--volume", help="volume folder name in output_dir (default: the config's render.name)")
    p.add_argument("--out", help="HTML path (default <output_dir>/qc/summary_<volume>.html)")
    p.add_argument("--flicker", action="append", default=[], metavar="DIR[:SERIES,...]",
                   help="zflicker output folder (and series) of this volume; repeatable")
    p.add_argument("--predict-flicker", action="store_true",
                   help="first measure the flicker the levels predict (tools/zflicker.py --config) and mark it as this volume's")
    p.add_argument("--chain", help="the chain's job list written by run_pipeline.sh: the page starts with the run's outcome")
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
         "render": read_json(step_path(cfg, "render", stem, "render.json")), "flicker": [flicker_arg(f, qc) for f in a.flicker],
         "plan": read_csv(step_path(cfg, "render", stem, "tiles.csv"), usecols=["z"])}
    if a.chain:
        R["chain"] = read_chain(a.chain) or []
    if a.predict_flicker and not R["flicker"] and R["own_config"]:
        got = predict_flicker(cfg, a.config, qc)
        if got:
            R["flicker"] = [got]
    out = Path(a.out) if a.out else qc / f"summary_{stem}.html"
    page = build(cfg, R, a.disk_size)
    atomic_write(out, lambda t: Path(t).write_text(page, encoding="utf-8"))
    print(f"qc_summary: wrote {out} ({len(page) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
