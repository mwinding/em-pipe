"""tools/split_stack.py: joined multi-day stacks split by the InLens days' slice counts, byte-identical pixels,
headers like the raw files, day order checked against InLens, manifest; inputs never changed."""
import hashlib
import importlib.util
import stat
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.imagej_tiff import ImageJStack  # noqa: E402

spec = importlib.util.spec_from_file_location("split_stack", ROOT / "tools" / "split_stack.py")
ss = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ss)

H, W = 64, 80


def scene(n, seed):
    """n slices of independent fine structure (each slice its own tissue)."""
    rng = np.random.default_rng(seed)
    return np.stack([ndimage.gaussian_filter(rng.standard_normal((H, W)), 1.0) for _ in range(n)])


def write(path, data, labels=None):
    md = {"unit": "nm", "spacing": 8.0}
    if labels is not None:
        md["Labels"] = labels
    tifffile.imwrite(path, data.astype(">u2"), byteorder=">", imagej=True, metadata=md,
                     resolution=((1, 8), (1, 8)))


@pytest.fixture
def dataset(tmp_path):
    """InLens days D12 (3 slices) and D13 (4), and the joined EsB stack of all 7, per tile (two tiles)."""
    raw, bs = tmp_path / "raw", tmp_path / "raw" / "Backscatter"
    bs.mkdir(parents=True)
    rng = np.random.default_rng(1)
    for t, tile in enumerate(("tile0-0", "tile0-1")):
        s = scene(7, seed=10 + t)
        inlens = 30000 + 3000 * s + 200 * rng.standard_normal(s.shape)
        esb = 20000 + 2000 * s + 400 * rng.standard_normal(s.shape)
        write(raw / f"M11_D12_{tile}.tif", inlens[:3])
        write(raw / f"M11_D13_{tile}.tif", inlens[3:])
        write(bs / f"M11_D12D13_ESB_{tile}.tif", esb, labels=[f"slice{i}" for i in range(7)])
    return raw, bs, tmp_path


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def test_run_splits_by_day_with_identical_pixels_and_manifest(dataset):
    raw, bs, tmp = dataset
    before = {p.name: md5(p) for p in list(raw.glob("*.tif")) + list(bs.glob("*.tif"))}
    manifest = tmp / "A_backscatter_split.md5"
    assert ss.main(["run", str(bs), str(raw), "--group-rw", "--manifest", str(manifest)]) == 0
    for tile in ("tile0-0", "tile0-1"):
        joined = ImageJStack(bs / f"M11_D12D13_ESB_{tile}.tif")
        for day, (a, n) in (("D12", (0, 3)), ("D13", (3, 4))):
            out = bs / f"M11_{day}_EsB_{tile}.tif"
            st = ImageJStack(out)
            assert st.n == n and st.contiguous and st.byteorder == ">"
            assert st.voxel_size_nm == joined.voxel_size_nm
            assert st.labels == [f"slice{i}" for i in range(a, a + n)]
            assert all(np.array_equal(st.read(i), joined.read(a + i)) for i in range(n))
            assert stat.S_IMODE(out.stat().st_mode) == 0o660
    lines = manifest.read_text().splitlines()
    assert len(lines) == 4 and all(md5(bs / ln.split("  ./")[1]) == ln.split()[0] for ln in lines)
    assert (tmp / "A_backscatter_split.md5.split_stack" / "check").is_dir()      # bookkeeping next to it
    assert {p.name: md5(p) for p in list(raw.glob("*.tif")) + list(bs.glob("M11_D12D13*.tif"))} == before


def test_slice_counts_must_add_up(dataset):
    raw, bs, _ = dataset
    write(raw / "M11_D13_tile0-1.tif", np.zeros((5, H, W)))       # 3 + 5 != 7
    with pytest.raises(SystemExit, match="7 slices, but the InLens days hold 3 \\+ 5"):
        ss.main(["plan", str(bs), str(raw)])


def test_days_in_the_wrong_order_fail_verify(tmp_path):
    raw, bs = tmp_path / "raw", tmp_path / "raw" / "Backscatter"
    bs.mkdir(parents=True)
    s = scene(6, seed=3)
    rng = np.random.default_rng(2)
    inlens = 30000 + 3000 * s + 200 * rng.standard_normal(s.shape)
    write(raw / "M11_D12_tile0-0.tif", inlens[3:])                  # the joined stack holds D13 first
    write(raw / "M11_D13_tile0-0.tif", inlens[:3])
    write(bs / "M11_D12D13_ESB_tile0-0.tif", 20000 + 2000 * s + 400 * rng.standard_normal(s.shape))
    assert ss.main(["run", str(bs), str(raw)]) == 1
    assert not (tmp_path / "raw" / "Backscatter_split.md5").exists()


def test_existing_split_is_left_alone(dataset, capsys):
    raw, bs, _ = dataset
    (bs / "M11_D12_EsB_tile0-0.tif").write_bytes(b"keep me")
    ss.main(["split", str(bs), str(raw), "--index", "0"])
    assert (bs / "M11_D12_EsB_tile0-0.tif").read_bytes() == b"keep me"
    assert "exists, left alone" in capsys.readouterr().out


def test_submit_dry_run_lists_the_chain(dataset, capsys):
    raw, bs, tmp = dataset
    assert ss.main(["submit", str(bs), str(raw), "--group-rw", "--manifest", str(tmp / "m.md5"), "--dry-run"]) == 0
    out = capsys.readouterr().out.splitlines()
    cmds = [ln for ln in out if ln.startswith("sbatch")]
    assert len(cmds) == 3 and "--array=0-3%6" in cmds[0] and "--group-rw" in cmds[0]
    assert all(f"--manifest {tmp / 'm.md5'}" in c for c in cmds)
