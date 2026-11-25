"""tools/qc_summary.py: the page on a synthetic run with parts missing, then with slice quality flags."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

import synth

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


qs, sq = load("qc_summary"), load("slice_quality")
SMALL = {
    "check": {"min_age_minutes": 0, "voxel_size_nm": [8, 8, 8]},
    "preview": {"factor": 4, "chunk_slices": 8},
    "stitch": {"sample_every": 5, "coarse_factor": 2, "fine_factor": 1, "fine_margin_px": 16},
    "align": {"scale": 1.0, "chunk_slices": 10, "max_points_per_pair": 40},
    "intensity": {"smooth_slices": 9, "balance_every": 3, "per_slice": False},
    "render": {"slab": 8, "chunk": [8, 32, 32], "shard_xy": 64, "num_scales": 1, "clahe": {"enabled": False}, "threads": 2},
}
SECTIONS = ("Dataset", "Processing settings", "Checks", "Slice quality flags", "Figures", "Files")


def step(cfg, name, *args):
    r = subprocess.run([sys.executable, "-m", f"pipeline.{name}", *args, "--config", str(cfg)], cwd=ROOT,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("qc_summary")
    truth = synth.make_dataset(tmp / "raw", n_slices=16, gaps={8: 3600})
    cfg = synth.write_config(tmp / "config.yaml", truth.raw_dir, tmp / "out", **SMALL)
    step(cfg, "check")
    for name, subs in (("preview", ("run", "merge")), ("stitch", ("run", "merge")), ("align", ("run", "solve"))):
        for sub in subs:
            step(cfg, name, sub)
    step(cfg, "intensity")
    step(cfg, "render", "init")
    step(cfg, "render", "run")
    return tmp, cfg


def check_page(page, tmp):
    for s in SECTIONS:
        assert f"<h2>{s}</h2>" in page, s
    assert str(tmp) not in page and "/private/" not in page and str(Path.home()) not in page    # only relative paths
    assert "<script" not in page and "http://" not in page and "https://" not in page          # nothing from the network


def test_page_with_parts_missing(run):
    tmp, cfg = run
    assert qs.main([str(cfg)]) == 0
    page = (tmp / "out" / "qc" / "summary_synthetic.html").read_text()
    check_page(page, tmp)
    assert "16 (z 0-15)" in page and "2 x 2" in page                      # dataset facts
    assert page.count('class="mark not-run"') >= 4                        # flicker, thickness, slice quality, seams
    assert "slurm/slice_quality.sbatch CONFIG" in page and "slurm/zflicker.sbatch" in page
    assert 'class="mark pass"' in page and "data:image/png;base64," in page
    assert "rendered slabs 2 of 2" in page


def test_page_with_slice_flags(run):
    tmp, cfg = run
    sq.main(["run", "--config", str(cfg), "--window", "64"])
    sq.main(["merge", "--config", str(cfg), "--window", "64"])
    assert qs.main([str(cfg), "--disk-size"]) == 0
    page = (tmp / "out" / "qc" / "summary_synthetic.html").read_text()
    check_page(page, tmp)
    assert 'href="slice_flags.csv"' in page and "slices carry a flag" in page and "GB on disk" in page
    assert "Slice quality metrics over z" in page


def test_flicker_is_marked_only_for_named_series(run):
    tmp, cfg = run
    d = tmp / "out" / "qc" / "zflicker_test"
    d.mkdir(exist_ok=True)
    series = {"raw (fixed window)": 11.0, "raw - windows": 0.06, "other volume": 6.4, "this volume": 0.07}
    (d / "zflicker.json").write_text(json.dumps({"series": {k: {"flicker_common_sd": v, "flicker_own_sd": 0.1, "planes": 16}
                                                              for k, v in series.items()}}))
    out = tmp / "page.html"
    qs.main([str(cfg), "--out", str(out)])
    page = out.read_text()
    assert "listed without a mark" in page and "other volume" in page
    qs.main([str(cfg), "--out", str(out), "--flicker", "zflicker_test:raw - windows,this volume"])
    page = out.read_text()
    row = page[page.index("<td>Flicker along z</td>"):]
    assert 'class="mark pass"' in row[:row.index("</span>") + 10] and "this volume (zflicker_test): 0.07 greys" in page
    qs.main([str(cfg), "--out", str(out), "--flicker", str(d)])          # every series but the raw tiles: 6.4 counts
    row = out.read_text()
    row = row[row.index("<td>Flicker along z</td>"):]
    assert 'class="mark attention"' in row[:row.index("</span>") + 10]
    check_page(out.read_text(), tmp)


def test_page_before_anything_has_run(tmp_path):
    cfg = synth.write_config(tmp_path / "config.yaml", tmp_path / "raw", tmp_path / "out")
    assert qs.main([str(cfg), "--out", str(tmp_path / "page.html")]) == 0
    page = (tmp_path / "page.html").read_text()
    check_page(page, tmp_path)
    assert "--steps check" in page
