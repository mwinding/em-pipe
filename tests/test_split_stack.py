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
    with pytest.raises(SystemExit, match="7 slices, but its InLens files hold 3 \\+ 5"):
        ss.main(["plan", str(bs), str(raw)])


def labels(times, det):
    """Slice labels as the microscope writes them (raw.label_pattern), one per time hhmmss."""
    return [f"SEM_26-09-21_{t:06d}_0-0-0_{det}_raw.tif" for t in times]


TIMES = [115, 2512, 4908, 102628, 110000, 120000, 235734]       # 3 slices in part1, 4 in part2


@pytest.fixture
def parts(tmp_path):
    """InLens day D21 in two parts per tile (3 + 4 slices) in raw/Approach, one EsB stack per tile holding both
    (raw/Backscatter, token ESB), and an ordinary EsB file of another day that must be left alone."""
    raw, app, bs = tmp_path / "raw", tmp_path / "raw" / "Approach", tmp_path / "raw" / "Backscatter"
    app.mkdir(parents=True)
    bs.mkdir()
    rng = np.random.default_rng(5)
    for t, tile in enumerate(("tile0-0", "tile1-1")):
        s = scene(7, seed=20 + t)
        inlens = 30000 + 3000 * s + 200 * rng.standard_normal(s.shape)
        write(app / f"M09_D21_{tile}_part1.tif", inlens[:3], labels(TIMES[:3], "InLens"))
        write(app / f"M09_D21_{tile}_part2.tif", inlens[3:], labels(TIMES[3:], "InLens"))
        write(bs / f"M09_D21_ESB_{tile}.tif", 20000 + 2000 * s + 400 * rng.standard_normal(s.shape),
              labels(TIMES, "ESB @ 600 V"))
    write(app / "M09_D20_tile0-0.tif", scene(2, seed=9) * 1000 + 30000)
    write(bs / "M09_D20_ESB_tile0-0.tif", scene(2, seed=9) * 1000 + 20000)
    return app, bs, tmp_path


def test_run_splits_the_parts_of_a_day_keeping_the_token(parts, capsys):
    app, bs, tmp = parts
    before = {p.name: md5(p) for p in list(app.glob("*.tif")) + list(bs.glob("*.tif"))}
    manifest = tmp / "A_backscatter_split.md5"
    assert ss.main(["run", str(bs), str(app), "--manifest", str(manifest)]) == 0
    out = capsys.readouterr().out
    assert "4 splits from 2 joined stacks" in out and "2 joined stacks match their InLens files' labels" in out
    assert "compared with the InLens file's, slice by slice: 4 of 4 splits" in out
    for tile in ("tile0-0", "tile1-1"):
        joined = ImageJStack(bs / f"M09_D21_ESB_{tile}.tif")
        for part, (a, n) in (("part1", (0, 3)), ("part2", (3, 4))):
            st = ImageJStack(bs / f"M09_D21_ESB_{tile}_{part}.tif")
            assert st.n == n and st.contiguous and st.byteorder == ">"
            assert st.labels == labels(TIMES[a:a + n], "ESB @ 600 V")
            assert all(np.array_equal(st.read(i), joined.read(a + i)) for i in range(n))
    assert not (bs / "M09_D20_ESB_tile0-0_part1.tif").exists()      # the ordinary EsB file is not a joined stack
    lines = manifest.read_text().splitlines()
    assert sorted(ln.split("  ./")[1] for ln in lines) == sorted(
        f"M09_D21_ESB_{t}_{p}.tif" for t in ("tile0-0", "tile1-1") for p in ("part1", "part2"))
    assert all(md5(bs / ln.split("  ./")[1]) == ln.split()[0] for ln in lines)
    assert {p.name: md5(p) for p in list(app.glob("*.tif")) + list(bs.glob("*.tif")) if p.name in before} == before
    # the splits next to the joined stacks are ordinary per-part EsB files: a later plan lists the same splits
    assert [Path(it["out"]).name for it in ss.plan(bs, app, bs, quiet=True)] == [
        f"M09_D21_ESB_{t}_{p}.tif" for t in ("tile0-0", "tile1-1") for p in ("part1", "part2")]


def test_parts_with_other_timestamps_stop_before_writing(parts):
    app, bs, _ = parts
    st = ImageJStack(app / "M09_D21_tile1-1_part2.tif")
    write(app / "M09_D21_tile1-1_part2.tif", np.stack([st.read(i) for i in range(st.n)]),
          labels([102628, 110000, 120001, 235734], "InLens"))
    with pytest.raises(SystemExit, match="1 slice labels differ .* first at slice 5"):
        ss.main(["split", str(bs), str(app)])
    assert not list(bs.glob("*_part*.tif"))


def test_parts_in_the_wrong_order_fail_verify(tmp_path):
    """Without labels, the slice order is checked on the images: part2's slices first in the joined stack."""
    app, bs = tmp_path / "Approach", tmp_path / "Backscatter"
    app.mkdir()
    bs.mkdir()
    s = scene(6, seed=4)
    rng = np.random.default_rng(3)
    inlens = 30000 + 3000 * s + 200 * rng.standard_normal(s.shape)
    write(app / "M09_D21_tile0-0_part1.tif", inlens[3:])
    write(app / "M09_D21_tile0-0_part2.tif", inlens[:3])
    write(bs / "M09_D21_ESB_tile0-0.tif", 20000 + 2000 * s + 400 * rng.standard_normal(s.shape))
    manifest = tmp_path / "m.md5"
    assert ss.main(["run", str(bs), str(app), "--manifest", str(manifest)]) == 1
    assert not manifest.exists()


def test_parts_must_be_numbered_from_one(parts):
    app, bs, _ = parts
    (app / "M09_D21_tile0-0_part2.tif").rename(app / "M09_D21_tile0-0_part3.tif")
    with pytest.raises(SystemExit, match="numbered 1, 3, not 1 to 2"):
        ss.main(["plan", str(bs), str(app)])


def test_a_day_of_a_multi_day_stack_in_parts(tmp_path):
    raw, bs = tmp_path / "raw", tmp_path / "raw" / "Backscatter"
    bs.mkdir(parents=True)
    s = scene(7, seed=6)
    rng = np.random.default_rng(4)
    inlens = 30000 + 3000 * s + 200 * rng.standard_normal(s.shape)
    write(raw / "M11_D12_tile0-0.tif", inlens[:2])
    write(raw / "M11_D13_tile0-0_part1.tif", inlens[2:4])
    write(raw / "M11_D13_tile0-0_part2.tif", inlens[4:])
    write(bs / "M11_D12D13_ESB_tile0-0.tif", 20000 + 2000 * s + 400 * rng.standard_normal(s.shape))
    assert ss.main(["run", str(bs), str(raw), "--manifest", str(tmp_path / "m.md5")]) == 0
    assert [ImageJStack(bs / n).n for n in ("M11_D12_EsB_tile0-0.tif", "M11_D13_EsB_tile0-0_part1.tif",
                                            "M11_D13_EsB_tile0-0_part2.tif")] == [2, 2, 3]


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
