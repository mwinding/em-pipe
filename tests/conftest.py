import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

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
