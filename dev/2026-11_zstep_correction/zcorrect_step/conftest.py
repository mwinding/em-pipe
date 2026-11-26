"""pytest setup for test_zcorrect.py in this folder, so it runs from dev/:

    python -m pytest dev/2026-11_zstep_correction/zcorrect_step/test_zcorrect.py

Puts this folder (zcorrect.py), tests/ (the synthetic data, synth.py) and the repository root (pipeline) on
sys.path, and gives the two fixtures the tests took from tests/conftest.py.
"""

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in (ROOT, ROOT / "tests", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import synth  # noqa: E402


@pytest.fixture(scope="session")
def synth_2x2(tmp_path_factory):
    """Clean 2×2 dataset, 24 slices crossing midnight (two daily files per tile)."""
    root = tmp_path_factory.mktemp("synth_2x2")
    return synth.make_dataset(root / "raw")


@pytest.fixture
def make_config(tmp_path):
    """make_config(truth_or_raw_dir, **sections) -> path to a YAML config with output in tmp_path."""
    def _make(raw, **sections):
        raw_dir = getattr(raw, "raw_dir", raw)
        return synth.write_config(tmp_path / "config.yaml", raw_dir, tmp_path / "out", **sections)
    return _make
