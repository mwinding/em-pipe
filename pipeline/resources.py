"""Slurm resources per job from the size of the dataset (``slurm.auto: true``).

Workflow tools size jobs from their inputs with a rule per job (Snakemake's ``resources:`` as functions of the
input size, Nextflow's dynamic directives); the FIB-SEM pipelines we looked at set them by hand per dataset.
The rules here are this pipeline's own: each job's array size, memory and time limit from the number of
slices, the tiles per slice and the tile size, with the constants taken from the hand-set values of three
datasets' configs and from Slurm's accounting of their runs (487 slices of 3 x 3 tiles of 81 Mpx; two volumes
of ~18,000 slices with 1 to 9 tiles of 80 to 180 Mpx). They are estimates with headroom, not limits known to
hold: a job that runs out of time or memory is re-submitted with an explicit setting, which always wins:

    slurm:
      auto: true
      render: {time: "36:00:00"}        # explicit keys override the automatic ones, key by key

What the rules use (the ``detected:`` section of the config, written by the ``em-pipe`` command from the raw
files' headers): slices, tiles (the most tiles in one slice), tile_px ([height, width] of the largest tile),
tile_images (tile slices in all) and raw_gpx (their pixels, in 10^9).

- array: one task per unit of work (a chunk of slices, a slab of planes) up to a cap per job.
- time: 3 x the expected time per task, in whole hours, at least 1 (the longest task of a run took up to 2.2 x
  the mean in the accounting).
- mem: a floor per job, more where a job holds whole tiles or the whole canvas in memory (refstats with
  decurtaining, render, export).
"""

import math

SAFETY = 3.0
# job: (cpus, floor of mem in GB, cap of the array or None for a single job)
JOBS = {
    "check": (4, 16, None), "preview": (2, 8, 40), "preview_merge": (2, 16, None),
    "refmask": (2, 8, 20), "refmask_merge": (2, 8, None),
    "stitch": (8, 32, 30), "stitch_merge": (2, 8, None), "align": (8, 16, 40), "align_solve": (4, 32, None),
    "shading": (2, 16, None), "overlaps": (2, 8, 40), "overlaps_merge": (2, 16, None), "patchgrid": (4, 16, None),
    "zrelations": (2, 8, 30), "zrelations_merge": (2, 16, None),
    "refstats": (4, 32, 40), "refstats_merge": (2, 8, None), "intensity": (2, 16, None),
    "render_init": (2, 16, None), "render": (8, 64, 30), "pyramid": (4, 8, 20), "export": (4, 24, None),
    "slice_quality": (2, 16, 40), "slice_quality_merge": (2, 32, None), "qc_summary": (2, 8, None),
    "precomputed_init": (2, 8, None), "precomputed": (4, 16, 40), "precomputed_verify": (2, 16, None),
    "upload": (4, 8, 16), "upload_finish": (2, 8, None),
}
# Expected task-seconds of a whole job per 10^9 raw pixels read (from the accounting, rounded up). stitch: its
# tasks are uneven (slices between two samples are stitched too where tiles move), the longest took 3 x the
# mean, so its rate is set from the longest task, not the mean
PER_GPX = {"preview": 11, "refmask": 8, "stitch": 60, "align": 31, "overlaps": 10, "zrelations": 4,
           "slice_quality": 6}
# ... per output plane of a 700 Mpx canvas
PER_PLANE = {"render": 18, "pyramid": 4, "precomputed": 70, "upload": 26}
DESTREAK_PER_GPX = 135       # render with decurtaining from a reference: added per 10^9 raw pixels
REFSTATS_PER_GPX = (220, 310)   # per 10^9 pixels of the sampled tile slices: plain, with decurtaining
FIXED_HOURS = {"check": 2, "preview_merge": 1, "refmask_merge": 1, "stitch_merge": 1, "align_solve": 2, "shading": 1,
               "overlaps_merge": 1, "patchgrid": 1, "zrelations_merge": 1, "refstats_merge": 1, "intensity": 1,
               "render_init": 1, "export": 2, "slice_quality_merge": 2, "qc_summary": 1,
               "precomputed_init": 1, "precomputed_verify": 2, "upload_finish": 2}
NEEDED = ("slices", "tiles", "tile_px", "tile_images", "raw_gpx")


def facts(shapes, groups):
    """The ``detected:`` numbers from the raw files' headers. shapes: {file: (slices, height, width)};
    groups: {file: (month, day, part)}, the files of one acquisition day and part being the tiles of the
    same slices."""
    days = {}
    for f, (n, h, w) in shapes.items():
        d = days.setdefault(groups[f], {"n": 0, "tiles": 0, "px": 0})
        d["n"], d["tiles"], d["px"] = max(d["n"], n), d["tiles"] + 1, d["px"] + h * w
    big = max(shapes.values(), key=lambda s: s[1] * s[2])
    return {"slices": int(sum(d["n"] for d in days.values())),
            "tiles": int(max(d["tiles"] for d in days.values())),
            "tile_px": [int(big[1]), int(big[2])],
            "tile_images": int(sum(n for n, _, _ in shapes.values())),
            "raw_gpx": _round(sum(n * h * w for n, h, w in shapes.values()) / 1e9),
            "canvas_mpx": _round(0.95 * max(d["px"] for d in days.values()) / 1e6)}


def _round(x):
    """One decimal for real datasets, six significant digits for very small ones (never 0)."""
    return round(x, 1) if x >= 10 else float(f"{x:.6g}")


def check_facts(f):
    """Message when ``detected:`` lacks what the rules need, else None."""
    def bad(v):
        v = min(v) if isinstance(v, (list, tuple)) and v else v
        return isinstance(v, bool) or not isinstance(v, (int, float)) or not v > 0

    missing = [k for k in NEEDED if not isinstance(f, dict) or bad(f.get(k))]
    if missing:
        return (f"slurm.auto needs the dataset's size in the detected: section ({', '.join(NEEDED)}); missing: "
                f"{', '.join(missing)}. The em-pipe command writes them; or set slurm.auto: false and the jobs' "
                "resources by hand")
    return None


def _hours(seconds):
    return f"{max(1, math.ceil(seconds / 3600.0)):02d}:00:00"


def _gb(x):
    return f"{int(math.ceil(x / 8.0) * 8)}G"


def auto(cfg):
    """{job: {array, cpus, mem, time}} for every job, from cfg['detected'] and the settings that change the
    work (decurtaining, refstats.every, chunk sizes, export.max_gb, intensity.balance_z_distance)."""
    from .config import get
    f = cfg.get("detected") or {}
    problem = check_facts(f)
    if problem:
        raise ValueError(problem)
    S, gpx = int(f["slices"]), float(f["raw_gpx"])
    tile_px = float(f["tile_px"][0]) * float(f["tile_px"][1])
    canvas = float(f.get("canvas_mpx") or 0.95 * f["tiles"] * tile_px / 1e6) / 700.0
    destreak = bool(get(cfg, "destreak.enabled", False))
    reference = destreak and get(cfg, "destreak.reference_from") is not None
    down = int(get(cfg, "render.downsample", 1) or 1)
    planes = math.ceil(S / down)
    every = int(get(cfg, "refstats.every", 32) or 32)
    samples = math.ceil(S / every) + 1
    units = {
        "preview": math.ceil(S / int(get(cfg, "preview.chunk_slices", 50))),
        "stitch": math.ceil(S / int(get(cfg, "stitch.sample_every", 50))),
        "align": math.ceil(S / int(get(cfg, "align.chunk_slices", 100))),
        "overlaps": math.ceil(S / int(get(cfg, "overlaps.chunk_slices", 100))),
        "zrelations": math.ceil(S / int(get(cfg, "zrelations.chunk_slices", 100))),
        "refstats": math.ceil(samples / int(get(cfg, "refstats.chunk_samples", 4))),
        "render": math.ceil(planes / int(get(cfg, "render.slab", 64))),
        "slice_quality": math.ceil(S / 50),
    }
    units["refmask"] = units["preview"]
    units["pyramid"] = units["precomputed"] = units["upload"] = max(2, math.ceil(S / 50))
    seconds = {job: k * gpx for job, k in PER_GPX.items()}
    for job, k in PER_PLANE.items():
        seconds[job] = k * planes * canvas / (down * down)
    if destreak:
        seconds["render"] += DESTREAK_PER_GPX * gpx * (1.0 if reference else 0.5)
    seconds["refstats"] = REFSTATS_PER_GPX[destreak] * samples * gpx / max(S, 1)
    out = {}
    for job, (cpus, mem, cap) in JOBS.items():
        o = {"cpus": cpus}
        if cap is not None:
            if job in ("refstats", "render") and destreak:
                cap *= 2          # decurtaining multiplies the work per task: more, shorter tasks
            o["array"] = max(1, min(units[job], cap))
            o["time"] = _hours(SAFETY * seconds[job] / o["array"])
        else:
            o["time"] = f"{FIXED_HOURS[job]:02d}:00:00"
        out[job] = o
        o["mem"] = f"{mem}G"
    tile_f32 = tile_px * 4 / 1e9                      # one tile as float32, GB
    if destreak:
        out["refstats"]["mem"] = _gb(max(32, 60 * tile_f32))
    # render: the slab buffer (64 planes of the canvas, 1 byte each) and the planes being rendered, ~128 bytes
    # per canvas pixel in the heaviest mode; 30 % headroom and 8 GB for the process
    out["render"]["mem"] = _gb(max(64, 1.3 * 128 * canvas * 700e6 / (down * down) / 1e9 + 8))
    out["export"]["mem"] = _gb(max(24, 2 * float(get(cfg, "export.max_gb", 4.0)) + 16))
    if S > 5000 and int(get(cfg, "intensity.balance_z_distance", 0) or 0) > 0:
        out["intensity"].update(mem="32G", time="02:00:00")
        out["zrelations_merge"]["mem"] = "16G"
    return out


def table(cfg, jobs=None):
    """The automatic values as text lines, one per job (for --dry-run output)."""
    a = auto(cfg)
    rows = [f"  {job:20s} " + "  ".join(f"{k} {a[job][k]}" for k in ("array", "cpus", "mem", "time") if k in a[job])
            for job in (jobs or a)]
    return rows
