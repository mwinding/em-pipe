"""Tests for the joint balance of tiles and slices: pipeline.zrelations (each tile against itself in the next
slices) and intensity.balance_z_distance (the gains and offsets of every (z, tile) solved together)."""

import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from scipy import ndimage

import synth
from pipeline import intensity, zrelations

X0 = 30000.0   # raw values of the synthetic tissue are around here


# ----- a synthetic run: tiles that jump independently, slices that jump together, slow drift -----

def truth_maps(n_slices, tiles, seed=0, tile_jump=500.0, slice_jump=250.0, gain_jump=0.01, drift=1500.0):
    """Per (z, tile) the map from true brightness x to the tile's corrected values u = a x + b.

    a = exp(log-gain), b = X0 (1 - a) + offset: each tile a constant gain and offset of its own, a jump of
    its own in every slice (offset tile_jump, log-gain gain_jump, robust SD), a jump every tile of the slice
    shares (slice_jump, gain_jump / 2), and a slow drift of the whole run (the offset from -drift to +drift
    along the run, the gain from -2 % to +2 %). Returns DataFrame z, tile, a, b, drift."""
    rng = np.random.default_rng(seed)
    rows = []
    shared = rng.normal(0, slice_jump, n_slices), rng.normal(0, gain_jump / 2, n_slices)
    const = {t: (rng.normal(0, 0.04), rng.normal(0, 600)) for t in tiles}
    for z in range(n_slices):
        slow = 2 * z / max(n_slices - 1, 1) - 1
        for t in tiles:
            lg = const[t][0] + 0.02 * slow + shared[1][z] + rng.normal(0, gain_jump)
            off = const[t][1] + drift * slow + shared[0][z] + rng.normal(0, tile_jump)
            a = np.exp(lg)
            rows.append({"z": z, "tile": t, "a": a, "b": X0 * (1 - a) + off, "drift": drift * slow})
    return pd.DataFrame(rows)


def relation(m, za, ta, zb, tb):
    """(gain, offset) with u_a = gain u_b + offset between two (z, tile) of the maps m (indexed by z, tile)."""
    (aa, ba), (ab, bb) = m.loc[(za, ta), ["a", "b"]], m.loc[(zb, tb), ["a", "b"]]
    gain = aa / ab
    return gain, ba - gain * bb


TILES3 = [f"{r}-{c}" for r in range(3) for c in range(3)]


def neighbours3():
    """Overlapping tile pairs of a 3 x 3 grid (sides and corners)."""
    out = []
    for i, a in enumerate(TILES3):
        for b in TILES3[i + 1:]:
            (ra, ca), (rb, cb) = (map(int, t.split("-")) for t in (a, b))
            if abs(ra - rb) <= 1 and abs(ca - cb) <= 1:
                out.append((a, b))
    return out


PCT = {"p0_5": -2600, "p1": -2200, "p50": 0, "p99": 2200, "p99_5": 2600,
       "t_p1": -1500, "t_p5": -900, "t_p25": -400, "t_p50": 0, "t_p75": 400, "t_p95": 900, "t_p99": 1500}


def write_run(out, m, n_slices, tiles, pairs, z_noise=3.0, seed=1):
    """slices.csv, preview/stats.csv, overlaps/relations.csv and zrelations/relations.csv of a stats-only run
    whose tiles map true brightness x to u = a x + b (maps m, indexed by z, tile). Tile t's tissue median is
    at x = X0 + 300 sin(z / 40 + t's number) (the content changes slowly), its percentiles PCT around it."""
    rng = np.random.default_rng(seed)
    t0 = datetime(2026, 11, 12, 16)
    slices, stats, rels, zrels = [], [], [], []
    for z in range(n_slices):
        for i, t in enumerate(tiles):
            r, c = map(int, t.split("-"))
            slices.append({"z": z, "timestamp": (t0 + timedelta(seconds=170 * z)).isoformat(), "tile": t,
                           "tile_row": r, "tile_col": c, "file": f"M11_D12_tile{t}.tif", "index": z, "height": 100,
                           "width": 100, "segment": 0, "seam": False, "excluded": False, "exclude_reason": "",
                           "label": ""})
            a, b = m.loc[(z, t), ["a", "b"]]
            x = X0 + 300 * np.sin(z / 40 + i)
            stats.append({"z": z, "tile": t, **{k: a * (x + d) + b for k, d in PCT.items()}, "tissue_fraction": 0.8})
        for a_, b_ in pairs:
            G, O = relation(m, z, a_, z, b_)
            mb = m.loc[(z, b_), "a"] * X0 + m.loc[(z, b_), "b"]
            rels.append({"z": z, "a": a_, "b": b_, "gain": G, "offset": O, "n": 10000, "n_total": 10000, "rms": 20.0,
                         "cnr": 50.0, "median_a": G * mb + O, "median_b": mb})
        if z + 1 < n_slices:
            for t in tiles:
                G, O = relation(m, z + 1, t, z, t)
                cb = m.loc[(z, t), "a"] * X0 + m.loc[(z, t), "b"]
                zrels.append({"z1": z, "z2": z + 1, "tile": t, "segment": 0, "distance": 1, "gain": G,
                              "offset": O + rng.normal(0, z_noise), "centre": cb, "n": 50000, "gain_raw": G,
                              "offset_raw": O, "centre_raw": cb})
    for d in ("check", "preview", "overlaps", "zrelations"):
        (out / "work" / d).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(slices).to_csv(out / "work" / "check" / "slices.csv", index=False)
    pd.DataFrame(stats).to_csv(out / "work" / "preview" / "stats.csv", index=False)
    pd.DataFrame(rels).to_csv(out / "work" / "overlaps" / "relations.csv", index=False)
    pd.DataFrame(zrels, columns=zrelations.COLUMNS).to_csv(out / "work" / "zrelations" / "relations.csv", index=False)
    (out / "work" / "zrelations" / "relations.json").write_text(json.dumps({"distance": 1, "shading": None,
                                                                            "patchgrid": None}))


def levels(tmp_path, out, name, **params):
    p = {"window": "median", "balance_source": "overlaps", "balance_smooth_slices": 1, "smooth_slices": 51,
         "output": name, **params}
    cfg = synth.write_config(tmp_path / f"{name}.yaml", tmp_path, out, intensity=p)
    assert intensity.main(["--config", str(cfg)]) == 0
    return pd.read_csv(out / "work" / "intensity" / name / "levels.csv", dtype={"tile": str}), cfg


def flicker(lv, m, stats):
    """(slice-wide flicker, spread between tiles) in greys: the tiles' tissue median as rendered (grey of
    t_p50 in its window) minus its running mean over 9 slices, the median over a slice's tiles (robust SD
    over z); and per slice the largest difference between tiles in the grey of the same true brightness."""
    d = lv.merge(stats[["z", "tile", "t_p50"]], on=["z", "tile"]).join(m[["a", "b"]], on=["z", "tile"])
    k = 255.0 / (d["hi"] - d["lo"])
    d["grey"] = (d["t_p50"] - d["lo"]) * k
    d["same"] = (d["a"] * X0 + d["b"] - d["lo"]) * k
    d = d.sort_values(["tile", "z"])
    d["hp"] = d["grey"] - d.groupby("tile")["grey"].transform(lambda s: s.rolling(9, center=True, min_periods=1).mean())
    common = d.groupby("z")["hp"].median()
    return (float(1.4826 * np.median(np.abs(common - common.median()))),
            float(d.groupby("z")["same"].agg(lambda s: s.max() - s.min()).max()), d)


def test_joint_balance_removes_the_slice_wide_flicker_the_per_slice_balance_leaves(tmp_path):
    """3 x 3 tiles jumping independently and together: the per-slice balance with one jump per slice leaves
    flicker the slice's tiles share; balance_z_distance 1 removes it, keeps the tiles matched and the tissue
    at its target grey, and its common window doesn't jump."""
    n_z = 300
    m = truth_maps(n_z, TILES3, seed=7).set_index(["z", "tile"])
    out = tmp_path / "out"
    write_run(out, m, n_z, TILES3, neighbours3())
    stats = pd.read_csv(out / "work" / "preview" / "stats.csv", dtype={"tile": str})
    now, _ = levels(tmp_path, out, "now", per_slice="slice")
    joint, cfg = levels(tmp_path, out, "joint", per_slice=False, balance_z_distance=1)
    f_now, spread_now, d_now = flicker(now, m, stats)
    f_joint, spread_joint, d = flicker(joint, m, stats)
    tissue = {k: v.groupby("z")["grey"].median() for k, v in (("now", d_now), ("joint", d))}
    print(f"slice-wide flicker: per-slice balance {f_now:.2f}, joint {f_joint:.2f} greys; tiles apart "
          f"{spread_now:.3f}, {spread_joint:.3f}; tissue grey range {np.ptp(tissue['now']):.1f}, "
          f"{np.ptp(tissue['joint']):.1f}")
    assert f_now > 1.0
    assert f_joint < 0.25 and f_joint < 0.05 * f_now
    assert spread_joint < 0.1                                   # tiles matched in every slice
    # The windows follow the drift (+-1500 raw, +-130 greys of the window): the tissue stays at its grey, apart
    # from the slow differences between the tiles' content (+-27 greys each)
    assert abs(tissue["joint"].median() - 140) < 1 and np.ptp(tissue["joint"]) < 20
    # The common window itself is smooth along z: it doesn't carry the jumps (the tiles' offsets do)
    df = intensity.balanced_scale(synth_cfg(cfg))
    w = intensity.common_window(df)["Lc"]
    assert (w - w.rolling(9, center=True, min_periods=1).mean()).abs().max() < 5.0       # raw units, 0.4 greys


def synth_cfg(path):
    from pipeline.config import load_config, step_defaults
    return load_config(path, step_defaults=step_defaults("intensity", "shading", "patchgrid"))


def test_joint_balance_off_leaves_the_levels_as_they_were(tmp_path):
    """balance_z_distance 0 (the default) is the per-slice balance, byte for byte, z relations or not."""
    n_z = 60
    m = truth_maps(n_z, TILES3, seed=2).set_index(["z", "tile"])
    out = tmp_path / "out"
    write_run(out, m, n_z, TILES3, neighbours3())
    levels(tmp_path, out, "default", per_slice="slice")
    levels(tmp_path, out, "zero", per_slice="slice", balance_z_distance=0)
    read = [(out / "work" / "intensity" / n / "levels.csv").read_bytes() for n in ("default", "zero")]
    assert read[0] == read[1]


@pytest.mark.parametrize("bad, match", [({"per_slice": "slice"}, "per_slice: false"),
                                        ({"per_slice": True}, "per_slice: false"),
                                        ({"balance_source": "thumbnails"}, "balance_source: overlaps"),
                                        ({"balance_z_distance": -1}, "balance_z_distance"),
                                        ({"balance_z_distance": 2}, "distance 1"),
                                        ({"balance_z_prior": 0}, "balance_z_prior")])
def test_joint_balance_settings_checked(tmp_path, bad, match):
    n_z = 12
    m = truth_maps(n_z, TILES3[:2], seed=2).set_index(["z", "tile"])
    out = tmp_path / "out"
    write_run(out, m, n_z, TILES3[:2], [("0-0", "0-1")])
    p = {"window": "median", "balance_source": "overlaps", "per_slice": False, "balance_z_distance": 1, **bad}
    cfg = synth.write_config(tmp_path / "c.yaml", tmp_path, out, intensity=p)
    with pytest.raises(ValueError, match=match):
        intensity.main(["--config", str(cfg)])


def test_joint_balance_needs_the_zrelations_step(tmp_path):
    n_z = 12
    m = truth_maps(n_z, TILES3[:2], seed=2).set_index(["z", "tile"])
    out = tmp_path / "out"
    write_run(out, m, n_z, TILES3[:2], [("0-0", "0-1")])
    (out / "work" / "zrelations" / "relations.csv").unlink()
    cfg = synth.write_config(tmp_path / "c.yaml", tmp_path, out, intensity={
        "window": "median", "balance_source": "overlaps", "per_slice": False, "balance_z_distance": 1})
    with pytest.raises(FileNotFoundError, match="zrelations"):
        intensity.main(["--config", str(cfg)])


# ----- zrelations: the measurement from the preview thumbnails -----

def write_thumbs(out, maps, field=None, shape=(150, 180), resin_cols=50, little=("1-1",), seed=4):
    """slices.csv, preview stats and thumbnails of tiles whose tissue (smooth texture, the same in every slice)
    shows as u = a x + b with maps {(z, tile): (a, b)}; resin (right ``resin_cols`` columns, plain, at a raw
    value that does not follow the tiles' jumps) is background. Tiles in ``little`` are nearly all resin. A
    flat-field ``field`` (thumbnail-sized) multiplies the raw values, and is written as the shading fields."""
    rng = np.random.default_rng(seed)
    h, w = shape
    x = ndimage.gaussian_filter(rng.standard_normal(shape), 1.2)
    x = X0 + 900 * x / x.std()
    zs = sorted({z for z, _ in maps})
    tiles = sorted({t for _, t in maps})
    t0 = datetime(2026, 11, 12, 16)
    slices, stats, index = [], [], []
    stacks = {t: [] for t in tiles}
    for z in zs:
        for t in tiles:
            a, b = maps[z, t]
            raw = a * x + b
            cols = w - 4 if t in little else resin_cols
            raw[:, w - cols:] = 41000.0 + rng.normal(0, 3, (h, cols))
            if field is not None:
                raw = raw * field
            lo, hi = np.percentile(raw, [0.5, 99.5])
            stacks[t].append(np.clip(np.rint((raw - lo) / (hi - lo) * 255), 0, 255).astype(np.uint8))
            r, c = map(int, t.split("-"))
            slices.append({"z": z, "timestamp": (t0 + timedelta(seconds=170 * z)).isoformat(), "tile": t,
                           "tile_row": r, "tile_col": c, "file": f"M11_D12_tile{t}.tif", "index": z,
                           "height": h * 16, "width": w * 16, "segment": 0, "seam": False, "excluded": False,
                           "exclude_reason": "", "label": ""})
            stats.append({"z": z, "tile": t, "p0_5": lo, "p99_5": hi})
            index.append({"z": z, "tile": t, "npy": f"z000000-{zs[-1] + 1:06d}_tile{t}.npy", "i": len(stacks[t]) - 1})
    (out / "work" / "check").mkdir(parents=True)
    (out / "work" / "preview" / "thumbs").mkdir(parents=True)
    pd.DataFrame(slices).to_csv(out / "work" / "check" / "slices.csv", index=False)
    pd.DataFrame(stats).to_csv(out / "work" / "preview" / "stats.csv", index=False)
    pd.DataFrame(index).to_csv(out / "work" / "preview" / "thumbs" / "index.csv", index=False)
    for t, st in stacks.items():
        np.save(out / "work" / "preview" / "thumbs" / f"z000000-{zs[-1] + 1:06d}_tile{t}.npy", np.stack(st))
    if field is not None:
        (out / "work" / "shading").mkdir(parents=True)
        np.savez(out / "work" / "shading" / "fields.npz", **{f"s0_t{t}": field.astype(np.float32) for t in tiles})
    return x


def run_zrelations(tmp_path, out, distance=1, **sections):
    cfg = synth.write_config(tmp_path / "z.yaml", tmp_path, out, intensity={"balance_z_distance": distance}, **sections)
    assert zrelations.main(["run", "--config", str(cfg)]) == 0
    assert zrelations.main(["merge", "--config", str(cfg)]) == 0
    return pd.read_csv(out / "work" / "zrelations" / "relations.csv", dtype={"tile": str}), cfg


def at_centre(rel, gain, offset):
    """How far a measured relation is from the true one (gain, offset), at the relation's centre (raw units)."""
    return (rel["gain"] * rel["centre"] + rel["offset"]) - (gain * rel["centre"] + offset)


def test_zrelations_match_each_tile_to_itself_in_the_next_slices(tmp_path):
    """The relation of a tile to itself d slices on is the change of its tissue (percentile matched, resin left
    out): recovered to well under a grey (~11 raw units) at its centre; tiles with too little tissue are left
    out, and distance 2 adds the slice after next. (Thumbnails of 20k tissue pixels: within ~3 raw units; real
    ones have ~10 times more.)"""
    maps = {}
    rng = np.random.default_rng(0)
    for z in range(4):
        for t in ("0-0", "0-1", "1-1"):
            a = np.exp(rng.normal(0, 0.02))
            maps[z, t] = (a, X0 * (1 - a) + rng.normal(0, 600))
    out = tmp_path / "out"
    write_thumbs(out, maps)
    rel, _ = run_zrelations(tmp_path, out, distance=2)
    assert set(rel["tile"]) == {"0-0", "0-1"}                    # 1-1: nearly all resin
    assert sorted(rel.loc[rel["tile"] == "0-0", "distance"]) == [1, 1, 1, 2, 2]
    for r in rel.itertuples():
        (a1, b1), (a2, b2) = maps[r.z1, r.tile], maps[r.z2, r.tile]
        gain, offset = a2 / a1, b2 - a2 / a1 * b1                # u_z2 = gain u_z1 + offset
        assert r.gain == pytest.approx(gain, rel=0.01)
        assert abs(at_centre(pd.Series(r._asdict()), gain, offset)) < 5.0
        assert r.n > 0.5 * 150 * 180
    assert (out / "qc" / "zrelations.png").stat().st_size > 0
    # Unchanged inputs: the chunks are current, run again skips them
    assert json.loads((out / "work" / "zrelations" / "relations.json").read_text())["distance"] == 2


def test_zrelations_are_on_the_flat_fielded_values(tmp_path):
    """With a flat-field the relation is the one between the corrected values u (raw / field), which the
    in-slice relations are on; on the raw thumbnails the offset is off by the field times the jump."""
    maps = {(z, "0-0"): (1.0, off) for z, off in enumerate([0.0, 900.0, -700.0])}
    h, w = 150, 180
    field = (1 + 0.25 * (np.arange(w) / (w - 1) - 0.5))[None, :] * np.ones((h, 1))
    out = tmp_path / "out"
    write_thumbs(out, maps, field=field, little=())
    rel, _ = run_zrelations(tmp_path, out, shading={"enabled": True})
    for r in rel.itertuples():
        offset = maps[r.z2, "0-0"][1] - maps[r.z1, "0-0"][1]
        on_u = (r.gain * r.centre + r.offset) - (r.centre + offset)
        on_raw = (r.gain_raw * r.centre_raw + r.offset_raw) - (r.centre_raw + offset)
        assert abs(on_u) < 3.0 and abs(on_raw) > 5 * max(abs(on_u), 1.0)


def test_joint_balance_from_images_end_to_end(tmp_path):
    """Synthetic tiles whose gains jump independently every slice (2 x 2, 3 %), through preview, overlaps,
    zrelations and intensity: the tiles' tissue as rendered flickers much less along z with the joint balance
    than with the per-slice one, and the seams are no worse."""
    import re

    import tifffile

    from pipeline import overlaps, transforms
    from synth import write_slices
    from test_intensity import overlap_diffs
    from test_preview import run_preview
    n = 40
    truth = synth.make_dataset(tmp_path / "raw", n_slices=n, overlap=(40, 48), tile_jitter=0.03, noise=0.002,
                               seed=11, z_positions=np.arange(n) * 0.15)
    out = tmp_path / "out"
    write_slices(truth, out)
    run_preview(synth.write_config(tmp_path / "preview.yaml", truth.raw_dir, out, preview={"factor": 2}))
    recs = [({"z": z, "tile": t, "segment": 0}, transforms.translation(*truth.tile_origin[t]))
            for z in range(n) for t in truth.tiles]
    (out / "work" / "stitch").mkdir()
    transforms.to_frame(recs).to_csv(out / "work" / "stitch" / "tiles.csv", index=False)
    ov = {"enabled": True, "factor": 2, "trim_px": 2, "smooth_px": 0.5, "min_pixels": 50, "chunk_slices": 8}
    base = {"window": "median", "balance_source": "overlaps", "balance_smooth_slices": 1, "smooth_slices": 11}
    cfg = synth.write_config(tmp_path / "z.yaml", truth.raw_dir, out, overlaps=ov, preview={"factor": 2},
                             intensity={**base, "per_slice": False, "balance_z_distance": 1})
    for step, cmds in ((overlaps, ("run", "merge")), (zrelations, ("run", "merge"))):
        for cmd in cmds:
            assert step.main([cmd, "--config", str(cfg)]) == 0
    planes = {}   # (z, tile) -> raw slice (a tile's slices can be in more than one file, one per day)
    for name, zs in truth.files.items():
        tile, stack = re.search(r"_tile(\d+-\d+)", name).group(1), tifffile.imread(truth.raw_dir / name).astype(float)
        planes.update({(z, tile): stack[i] for i, z in enumerate(zs)})
    result = {}
    for name, params in (("now", {"per_slice": "slice"}), ("joint", {"per_slice": False, "balance_z_distance": 1})):
        c = synth.write_config(tmp_path / f"{name}.yaml", truth.raw_dir, out, overlaps=ov,
                               intensity={**base, **params, "output": name})
        assert intensity.main(["--config", str(c)]) == 0
        lv = pd.read_csv(out / "work" / "intensity" / name / "levels.csv", dtype={"tile": str})
        grey = {(r.z, r.tile): np.median((planes[r.z, r.tile] - r.lo) / (r.hi - r.lo) * 255) for r in lv.itertuples()}
        g = pd.Series(grey).rename_axis(["z", "tile"]).reset_index(name="g").sort_values(["tile", "z"])
        g["hp"] = g["g"] - g.groupby("tile")["g"].transform(lambda s: s.rolling(9, center=True, min_periods=1).mean())
        common = g.groupby("z")["hp"].median()
        result[name] = (float(1.4826 * np.median(np.abs(common - common.median()))),
                        float(np.median(overlap_diffs(truth, lv))))
    print("rendered tissue flicker (greys) / median seam difference: per-slice %.2f / %.2f, joint %.2f / %.2f"
          % (*result["now"], *result["joint"]))
    assert result["joint"][0] < 0.25 * result["now"][0]
    assert result["joint"][1] < 1.2 * result["now"][1] + 0.2


def test_joint_solve_removes_fast_jumps_keeps_drift_and_tile_matches():
    """joint_solve on exact relations: within a slice every tile maps the same true brightness to the same
    common value; along z the jumps (each tile's and the slice's) are gone, the slow drift is still there."""
    n_z = 600
    m = truth_maps(n_z, TILES3, seed=3).set_index(["z", "tile"])
    nodes = [(z, t) for z in range(n_z) for t in TILES3]
    k = {nd: i for i, nd in enumerate(nodes)}
    ia, ib, gain, off, w, cen = [], [], [], [], [], []
    rng = np.random.default_rng(5)
    for z in range(n_z):
        for a, b in neighbours3():
            G, O = relation(m, z, a, z, b)
            ia.append(k[z, a]), ib.append(k[z, b]), gain.append(G), off.append(O), w.append(400.0)
            cen.append(m.loc[(z, b), "a"] * X0 + m.loc[(z, b), "b"])
        if z + 1 < n_z:
            for t in TILES3:   # the tissue changes a little from slice to slice: 15 raw units at the centre
                G, O = relation(m, z + 1, t, z, t)
                ia.append(k[z + 1, t]), ib.append(k[z, t]), gain.append(G), off.append(O + rng.normal(0, 15))
                w.append(40.0), cen.append(m.loc[(z, t), "a"] * X0 + m.loc[(z, t), "b"])
    w = np.array(w)
    g, o = intensity.joint_solve(len(nodes), ia, ib, gain, off, w, w, cen, 0.4, 0.4, X0)
    a, b = m["a"].to_numpy(), m["b"].to_numpy()        # in the order of nodes
    common = g * (a * X0 + b) + o                       # where each node puts the true brightness X0
    per = pd.DataFrame({"z": [nd[0] for nd in nodes], "c": common})
    spread = per.groupby("z")["c"].agg(lambda s: s.max() - s.min())
    assert spread.max() < 1.0                           # tiles matched (exact in-slice relations)
    level = per.groupby("z")["c"].mean()
    raw = pd.Series(a * X0 + b).groupby(per["z"]).mean()   # the raw tiles' mean level, jumps and all

    def fast(s):
        return s - s.rolling(9, center=True, min_periods=1).mean()
    assert fast(raw).std() > 200                        # the jumps the solve has to take out
    assert fast(level).std() < 0.05 * fast(raw).std()
    # The slow drift stays (on the common scale; the windows follow it): away from the run's ends
    drift = m.groupby("z")["drift"].first()
    slow = level.rolling(51, center=True).mean() - level.mean()
    inner = slice(100, n_z - 100)
    assert np.corrcoef(slow[inner], drift[inner])[0, 1] > 0.98
    assert (slow[inner] - drift[inner]).abs().max() < 0.15 * drift.abs().max()


def test_joint_solve_scales_to_a_long_run():
    """18,000 slices x 4 tiles (a 35h-sized block) in one sparse solve: tiles matched, jumps gone."""
    n_z, n_t = 18000, 4
    rng = np.random.default_rng(9)
    la = rng.normal(0, 0.01, (n_z, n_t)) + rng.normal(0, 0.005, (n_z, 1))
    off = rng.normal(0, 500, (n_z, n_t)) + rng.normal(0, 250, (n_z, 1))
    a = np.exp(la).ravel()
    b = (X0 * (1 - np.exp(la)) + off).ravel()
    node = np.arange(n_z * n_t).reshape(n_z, n_t)
    pairs = [(0, 1), (0, 2), (1, 3), (2, 3)]
    ia = np.concatenate([node[:, p] for p, _ in pairs] + [node[1:, t] for t in range(n_t)])
    ib = np.concatenate([node[:, q] for _, q in pairs] + [node[:-1, t] for t in range(n_t)])
    gain = a[ia] / a[ib]
    offset = b[ia] - gain * b[ib]
    centre = a[ib] * X0 + b[ib]
    inslice = np.arange(len(ia)) < n_z * len(pairs)
    offset[~inslice] += rng.normal(0, 3, (~inslice).sum())
    w = np.where(inslice, 400.0, 40.0)
    g, o = intensity.joint_solve(n_z * n_t, ia, ib, gain, offset, w, w, centre, 0.4, 0.4, X0)
    common = (g * (a * X0 + b) + o).reshape(n_z, n_t)
    assert np.ptp(common, axis=1).max() < 1.0
    level = pd.Series(common.mean(axis=1))
    raw = pd.Series((a * X0 + b).reshape(n_z, n_t).mean(axis=1))
    fast = [s - s.rolling(9, center=True, min_periods=1).mean() for s in (level, raw)]
    assert fast[0].std() < 0.05 * fast[1].std()


# ----- run_pipeline.sh -----

from test_run_pipeline import calls, env, names, run, write_cfg  # noqa: E402,F401  (env: the fixture)


def test_zrelations_step_when_the_joint_balance_is_on(tmp_path, env):  # noqa: F811
    """zrelations: an array run and a merge after patchgrid and before refstats and intensity, only with
    intensity.balance_z_distance > 0."""
    env, log = env
    cfg = write_cfg(tmp_path, overlaps={"enabled": True, "patch_grid": 8}, patchgrid={"enabled": True},
                    refstats={"enabled": True}, intensity={"balance_source": "overlaps", "per_slice": False,
                                                           "balance_z_distance": 1},
                    slurm_extra={"zrelations": {"array": 3, "time": "00:30:00"}})
    jobs = calls(run(cfg, env, "--from", "patchgrid", "--dry-run").stdout.splitlines())
    assert [j["job-name"] for j in jobs][:6] == ["em-patchgrid", "em-zrelations-run", "em-zrelations-merge",
                                                 "em-refstats-run", "em-refstats-merge", "em-intensity"]
    zr = jobs[1]
    assert zr["script"] == "slurm/zrelations.sbatch" and zr["args"][1:] == ["run"] and zr["array"] == "0-2"
    assert zr["time"] == "00:30:00" and zr["dependency"] == "afterok:<em-patchgrid>"
    assert jobs[2]["args"][1:] == ["merge"] and jobs[3]["dependency"] == "afterok:<em-zrelations-merge>"
    p = run(write_cfg(tmp_path, overlaps={"enabled": True}), env, "--steps", "zrelations,intensity", "--dry-run")
    assert names(p.stdout) == ["em-intensity"] and "balance_z_distance is 0: zrelations skipped" in p.stderr
