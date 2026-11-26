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
SECTIONS = ("Dataset", "Processing settings", "Checks", "Slice quality flags", "Figures", "Files", "The run",
            "Decisions for a person")


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


# ----- the run's outcome: where a chain stopped, the verdict, the decisions left to a person ---------------------

SACCT = """101|COMPLETED|0:0|12|48
102_0|COMPLETED|0:0|300|600
102_1|COMPLETED|0:0|420|840
103|FAILED|1:0|30|60
104_[0-9]|CANCELLED|0:0|0|0
105|CANCELLED by 0|0:0|0|0
"""
CHAIN = [("em-check", "101"), ("em-preview-run", "102"), ("em-preview-merge", "103"), ("em-stitch-run", "104"),
         ("em-stitch-merge", "105")]


def test_job_states_from_slurm_accounting():
    jobs = qs.job_states(CHAIN, SACCT)
    assert [j["state"] for j in jobs] == ["COMPLETED", "COMPLETED", "FAILED", "CANCELLED", "CANCELLED"]
    assert jobs[1] == {"name": "em-preview-run", "id": "102", "state": "COMPLETED", "tasks": 2, "elapsed_s": 420, "cpu_s": 1440}
    # an array with one failed task is failed; a job the accounting does not list is unknown
    assert qs.job_states([("a", "7")], "7_0|COMPLETED|0:0|5|5\n7_1|TIMEOUT|0:0|9|9\n")[0]["state"] == "TIMEOUT"
    assert qs.job_states([("a", "8")], "")[0]["state"] == "unknown"


def test_page_says_where_the_chain_stopped_and_why(run):
    tmp, cfg = run
    logs = tmp / "out" / "work" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "em-preview-merge-103.out").write_text("2026-01-01 00:00:00 INFO start\n"
                                                   "2026-01-01 00:00:01 ERROR pipeline.preview: 3 chunks missing: run preview\n")
    chain = tmp / "chain.tsv"
    chain.write_text("".join(f"{n}\t{i}\n" for n, i in CHAIN))
    assert qs.read_chain(chain) == CHAIN
    R = {"chain": CHAIN, "sacct": SACCT}
    html, stopped = qs.run_section(qs.load_config(str(cfg), None, None, qs.step_defaults()), R)
    assert stopped == ("The chain stopped at em-preview-merge (job 103, FAILED). 2 later jobs did not run: "
                       "em-stitch-run, em-stitch-merge.")
    assert "3 chunks missing: run preview" in html and "core-hours" in html
    cls, word, detail = qs.verdict([("Stitch", "", "", qs.PASS, "")], stopped)
    assert (cls, word) == ("stopped", "STOPPED") and "em-preview-merge" in detail
    # a chain that ran through: no stop
    ok = "".join(f"{i}|COMPLETED|0:0|10|20\n" for _, i in CHAIN)
    assert qs.run_section(qs.load_config(str(cfg), None, None, qs.step_defaults()), {"chain": CHAIN, "sacct": ok})[1] is None


def test_verdict_and_decisions():
    rows = [("Stitch", "", "", qs.PASS, ""), ("Align", "", "", qs.PASS, ""), ("Flicker along z", "", "", qs.NOTRUN, "")]
    cls, word, detail = qs.verdict(rows, None)
    assert (cls, word) == ("pass", "PASSED") and "Every check that ran passed (2)" in detail and "Flicker along z" in detail
    cls, word, detail = qs.verdict(rows + [("Tile jumps", "", "", qs.ATTN, "")], None)
    assert (cls, word) == ("attention", "ATTENTION") and "1 check to look at: Tile jumps" in detail
    html = qs.decisions_section(rows + [("Tile jumps", "", "", qs.ATTN, "")])
    for words in ("Leaving out files or slices", "reference-tissue mask", "Judging the result", "Uploading", "Deleting"):
        assert words in html
    assert "Tile jumps" in html


def test_page_with_the_chain_and_the_predicted_flicker(run):
    """As the chain's last job writes it: the levels' predicted flicker measured and marked, the verdict on top."""
    tmp, cfg = run
    chain = tmp / "chain.tsv"
    chain.write_text("em-check\t101\n")
    out = tmp / "chain_page.html"
    assert qs.main([str(cfg), "--out", str(out), "--chain", str(chain), "--predict-flicker"]) == 0
    page = out.read_text()
    check_page(page, tmp)
    assert 'class="verdict ' in page and "raw - windows (zflicker_levels)" in page
    assert (tmp / "out" / "qc" / "zflicker_levels" / "zflicker.json").exists()
    assert "em-check" in page and "Slurm id" in page
