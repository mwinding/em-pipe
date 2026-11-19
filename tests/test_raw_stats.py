"""tools/raw_stats.py: classification of a synthetic tile, histogram bookkeeping, run / merge / compare."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import ndimage

import synth

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("raw_stats", ROOT / "tools" / "raw_stats.py")
raw_stats = importlib.util.module_from_spec(spec)
spec.loader.exec_module(raw_stats)


def synthetic_tile(h=2048, w=2048, resin_from=0.6, seed=0):
    """Textured 'tissue' on the left, smooth bright 'resin' on the right (touching the tile border)."""
    rng = np.random.default_rng(seed)
    small = ndimage.gaussian_filter(rng.standard_normal((-(-h // 16), -(-w // 16))), 1.0)
    tissue = 33000 + 2500 * small / small.std()
    img = np.kron(tissue, np.ones((16, 16)))[:h, :w]
    img += rng.normal(0, 150, img.shape)
    x0 = int(w * resin_from)
    img[:, x0:] = 39000 + rng.normal(0, 150, (h, w - x0))
    return np.clip(img, 0, 65535).astype(np.uint16), x0


def test_classify_finds_resin_and_tissue():
    img, x0 = synthetic_tile()
    small, _ = raw_stats.downsample(img, 16)
    lab, strict = raw_stats.classify(small)
    xs = x0 // 16
    assert (lab[:, : xs - 4] == raw_stats.TISSUE).mean() > 0.95
    assert (lab[:, xs + 4:] == raw_stats.RESIN).mean() > 0.95
    assert strict.any() and not (strict & (lab != raw_stats.RESIN)).any()      # strict is a subset of resin
    assert not strict[:16].any() and not strict[:, -16:].any()                  # tile-edge band left out


def test_tile_histograms_account_for_every_pixel():
    img, _ = synthetic_tile(h=1536, w=1560, resin_from=0.5)
    hists, fracs = raw_stats.tile_histograms(img, 16)
    crop = img[: 1536 // 16 * 16, : 1560 // 16 * 16]
    assert sum(int(hists[k].sum()) for k in ("tissue", "resin", "offsample")) == crop.size
    assert hists["resin_strict"].sum() <= hists["resin"].sum()
    med = raw_stats.percentiles(hists["resin"], (50,))[0]
    assert 38500 < med < 39500
    assert abs(sum(fracs[k] for k in ("tissue", "resin", "offsample")) - 1) < 1e-9


def test_percentiles_of_histogram():
    h = np.zeros(10, np.int64)
    h[[2, 5, 7]] = [1, 2, 1]
    assert raw_stats.percentiles(h, (25, 50, 100)) == [2.0, 5.0, 7.0]
    assert np.isnan(raw_stats.percentiles(np.zeros(4, np.int64), (50,))[0])


@pytest.fixture
def small_dataset(tmp_path):
    truth = synth.make_dataset(tmp_path / "raw", n_slices=6, tile_shape=(256, 288))
    out = tmp_path / "out"
    synth.write_slices(truth, out)
    cfg = synth.write_config(tmp_path / "config.yaml", truth.raw_dir, out)
    return truth, cfg, out


def test_run_writes_results(small_dataset):
    truth, cfg, out = small_dataset
    raw_stats.main(["run", "--config", str(cfg), "--slices", "3", "--factor", "4"])
    d = out / "qc" / "raw_stats"
    for name in ("hist.npz", "tiles.csv", "pooled.png", "tiles.png", "meta.json"):
        assert (d / name).exists(), name
    tiles = pd.read_csv(d / "tiles.csv", dtype={"tile": str})
    assert len(tiles) == 3 * len(truth.tiles)
    meta = json.loads((d / "meta.json").read_text())
    assert len(meta["z"]) == 3 and meta["factor"] == 4
    with np.load(d / "hist.npz") as h:
        counted = sum(int(h[k].sum()) for k in ("tissue", "resin", "offsample"))
    assert counted == 3 * len(truth.tiles) * (256 // 4 * 4) * (288 // 4 * 4)


def test_tasks_then_merge_match_single_run(small_dataset, tmp_path):
    _, cfg, out = small_dataset
    single, parts = tmp_path / "single", tmp_path / "parts"
    raw_stats.main(["run", "--config", str(cfg), "--slices", "4", "--factor", "4", "--out", str(single)])
    for k in range(2):
        raw_stats.main(["run", "--config", str(cfg), "--slices", "4", "--factor", "4", "--out", str(parts),
                        "--task-id", str(k), "--num-tasks", "2"])
    raw_stats.main(["merge", "--config", str(cfg), "--out", str(parts)])
    with np.load(single / "hist.npz") as a, np.load(parts / "hist.npz") as b:
        for k in a.files:
            assert np.array_equal(a[k], b[k]), k
    ta = pd.read_csv(single / "tiles.csv", dtype={"tile": str})
    tb = pd.read_csv(parts / "tiles.csv", dtype={"tile": str})
    pd.testing.assert_frame_equal(ta.sort_values(["z", "tile"]).reset_index(drop=True),
                                  tb.sort_values(["z", "tile"]).reset_index(drop=True))
    raw_stats.main(["compare", "--out", str(tmp_path / "cmp"), f"A={single}", f"B={parts}"])
    assert (tmp_path / "cmp" / "compare_pooled.png").exists() and (tmp_path / "cmp" / "compare_tiles.png").exists()


def test_merge_refuses_incomplete_parts(small_dataset, tmp_path):
    _, cfg, _ = small_dataset
    parts = tmp_path / "parts"
    raw_stats.main(["run", "--config", str(cfg), "--slices", "4", "--factor", "4", "--out", str(parts),
                    "--task-id", "0", "--num-tasks", "2"])
    with pytest.raises(SystemExit):
        raw_stats.main(["merge", "--config", str(cfg), "--out", str(parts)])
