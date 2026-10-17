"""overlaps: brightness relations of overlapping tiles from the raw tiles, and intensity's balance from them."""

import numpy as np
import pandas as pd
import pytest

import synth
from pipeline import intensity, overlaps, shading, transforms
from pipeline.config import load_config
from synth import write_slices
from test_intensity import overlap_diffs
from test_preview import run_preview

N = 24
JITTER = 0.06   # each tile's gain wanders independently by ~6 % from slice to slice
OVERLAPS = {"enabled": True, "factor": 2, "trim_px": 2, "smooth_px": 0.5, "min_pixels": 50, "chunk_slices": 8}


def test_fit_relation_is_robust_and_symmetric():
    rng = np.random.default_rng(1)
    truth = rng.uniform(30000, 40000, 5000)
    vb = truth + rng.normal(0, 150, truth.size)
    va = 1.12 * truth - 2500 + rng.normal(0, 150 * 1.12, truth.size)
    va[:400] += 6000                                   # a streak in tile a only
    g, o, n, rms, cnr = overlaps.fit_relation(va, vb)
    assert g == pytest.approx(1.12, abs=0.01) and o == pytest.approx(-2500, abs=400)
    assert n < 4700 and rms < 400 and cnr > 20
    flat = overlaps.fit_relation(36000 + rng.normal(0, 150, 3000), 35000 + rng.normal(0, 150, 3000))
    assert flat is not None and flat[4] < 5        # resin only: spread ~ noise, the gain is barely pinned
    # an ordinary regression on these noisy pairs would flatten the gain; the fit is symmetric
    g_ba = overlaps.fit_relation(vb, va)[0]
    assert g * g_ba == pytest.approx(1, abs=0.01)
    assert overlaps.fit_relation(va[:100], vb[:100]) is None                # too few pairs
    assert overlaps.fit_relation(np.full(600, 5.0), np.full(600, 7.0)) is None   # no spread


@pytest.fixture(scope="module")
def jittered(tmp_path_factory):
    """Dataset whose tiles change brightness independently every slice; preview, ground-truth stitch."""
    root = tmp_path_factory.mktemp("overlaps")
    truth = synth.make_dataset(root / "raw", n_slices=N, overlap=(40, 48), tile_jitter=JITTER, noise=0.002, seed=11)
    out = root / "out"
    write_slices(truth, out)
    run_preview(synth.write_config(root / "preview.yaml", truth.raw_dir, out, preview={"factor": 4}))
    recs = [({"z": z, "tile": t, "segment": 0}, transforms.translation(*truth.tile_origin[t]))
            for z in range(N) for t in truth.tiles]
    (out / "work" / "stitch").mkdir()
    transforms.to_frame(recs).to_csv(out / "work" / "stitch" / "tiles.csv", index=False)
    return truth, root, out


def _config(root, truth, out, name, **intensity_params):
    params = {"window": "percentile", "per_slice": False, **intensity_params}
    return synth.write_config(root / f"{name}.yaml", truth.raw_dir, out, intensity=params, overlaps=OVERLAPS)


def test_relations_recover_each_slices_tile_gains(jittered):
    truth, root, out = jittered
    cfg = _config(root, truth, out, "ov")
    for task in range(2):
        assert overlaps.main(["run", "--config", str(cfg), "--task-id", str(task), "--num-tasks", "2"]) == 0
    assert overlaps.main(["merge", "--config", str(cfg)]) == 0
    rel = pd.read_csv(out / "work" / "overlaps" / "relations.csv", dtype={"a": str, "b": str})
    # 2x2 tiles: 4 edge pairs and 2 diagonal pairs (corner overlaps) per slice
    assert rel.groupby("z").size().max() == 6 and rel["z"].nunique() == N
    # raw = lo + jitter * content * (hi - lo): gain a/b is the ratio of the two tiles' jitters
    want = np.array([truth.jitter[r.z][r.a] / truth.jitter[r.z][r.b] for r in rel.itertuples()])
    edges = rel["n"] > 300
    assert np.abs(rel["gain"][edges] / want[edges] - 1).max() < 0.02
    assert (out / "qc" / "overlaps.png").exists()
    # a re-run measures nothing again; a changed setting re-measures
    chunk = next((out / "work" / "overlaps" / "chunks").glob("chunk_*.csv"))
    stamp = chunk.stat().st_mtime_ns
    assert overlaps.main(["run", "--config", str(cfg)]) == 0
    assert chunk.stat().st_mtime_ns == stamp
    cfg2 = synth.write_config(root / "ov2.yaml", truth.raw_dir, out, overlaps={**OVERLAPS, "reject": 2.5})
    with pytest.raises(RuntimeError, match="out of date"):
        overlaps.main(["merge", "--config", str(cfg2)])


def test_balance_from_overlaps_follows_every_slice(jittered):
    """Thumbnail balance (every 10 slices, smoothed) can't follow tiles that change every slice;
    the overlap relations, solved per slice, match the tiles in every slice."""
    truth, root, out = jittered
    cfg = _config(root, truth, out, "ov")
    if not (out / "work" / "overlaps" / "relations.csv").exists():
        assert overlaps.main(["run", "--config", str(cfg)]) == 0
        assert overlaps.main(["merge", "--config", str(cfg)]) == 0
    levels = {}
    for name, params in {"thumbs": {}, "overlaps": {"balance_source": "overlaps", "balance_smooth_slices": 1}}.items():
        c = _config(root, truth, out, name, **params)
        assert intensity.main(["--config", str(c)]) == 0
        levels[name] = pd.read_csv(out / "work" / "intensity" / "levels.csv", dtype={"tile": str})
    d_thumbs, d_overlaps = overlap_diffs(truth, levels["thumbs"]), overlap_diffs(truth, levels["overlaps"])
    print("median seam |difference| (greys): thumbnails %.1f, overlaps %.1f" % (np.median(d_thumbs), np.median(d_overlaps)))
    assert np.median(d_overlaps) < 0.4 * np.median(d_thumbs) and np.median(d_overlaps) < 3


def test_per_slice_modes(jittered):
    truth, root, out = jittered
    for mode in (True, "slice", False):
        c = _config(root, truth, out, "ps", per_slice=mode)
        assert intensity.main(["--config", str(c)]) == 0
    with pytest.raises(ValueError, match="true, false or slice"):
        intensity.main(["--config", str(_config(root, truth, out, "bad", per_slice="tile"))])
    c = synth.write_config(root / "none.yaml", truth.raw_dir, root / "empty", intensity={"balance_source": "overlaps"})
    with pytest.raises(FileNotFoundError, match="overlaps step"):
        intensity.overlap_measurements(load_config(c, step_defaults=intensity.DEFAULTS))


def test_solve_weights_the_gain_of_low_contrast_overlaps():
    # a-b says gain 1.2 (well measured), a-c and b-c say otherwise but with gain weight ~0
    rels = [("a", "b", 1.2, 0.0, 10000, 1.0), ("a", "c", 0.5, 0.0, 10000, 1e-6), ("b", "c", 3.0, 0.0, 10000, 1e-6)]
    g = intensity.solve(rels)
    assert g["a"][0] * 1.2 == pytest.approx(g["b"][0], rel=1e-3)
    assert shading.DEFAULTS["shading"]["enabled"] is False


def test_solve_matches_tiles_at_the_data_not_at_zero():
    """Raw EM sits far from 0 (~37000): a 3 % gain error moves a relation's value at 0 by ~1100,
    so offsets matched at 0 mismatch the tiles at their data; matched at the relation's centre they don't."""
    true_o = {"a": 0.0, "b": 400.0, "c": -300.0}       # b = a + 400, c = a - 300 (gains all 1)
    m = 37000.0
    rels = []
    for (x, y), err in ((("a", "b"), 1.03), (("b", "c"), 0.97), (("a", "c"), 1.02)):
        # v_x ~ err * v_y + off, exact at v_y = m: v_x(m) = m - true_o[y] + true_o[x]
        off = (m - true_o[y] + true_o[x]) - err * m
        rels.append((x, y, err, off, 10000, 1e-9, m))      # gains barely trusted, as for noisy overlaps
    sol = intensity.solve(rels)
    for x, y, err, off, *_ in rels:
        vx = err * m + off
        mismatch = (sol[x][0] * vx + sol[x][1]) - (sol[y][0] * m + sol[y][1])
        assert abs(mismatch) < 5
    at_zero = intensity.solve([r[:6] for r in rels])
    x, y, err, off, *_ = rels[0]
    assert abs((at_zero[x][0] * (err * m + off) + at_zero[x][1]) - (at_zero[y][0] * m + at_zero[y][1])) > 300
