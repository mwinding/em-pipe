"""tools/bucket_list.py: folders with object counts and sizes, and which hold a complete precomputed volume;
read-only (tested on a local folder through file://)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import bucket_list as bl  # noqa: E402


def _tree(root):
    files = {"volA/info": 10, "volA/provenance": 5, "volA/8_8_8/0.shard": 100, "volA/16_16_16/0.shard": 50,
             "volB/info": 10, "volB/8_8_8/0.shard": 70, "loose.txt": 3}
    for rel, size in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x" * size)
    return files


def test_summary_by_folder_and_volume_state(tmp_path):
    files = _tree(tmp_path)
    rows = bl.summarise(bl.objects(f"file://{tmp_path}"))
    assert rows["volA"] == [4, 165, True, True]           # complete: info and provenance at its top
    assert rows["volB"] == [2, 80, True, False]           # info only
    assert rows["."] == [1, 3, False, False]
    assert sum(r[0] for r in rows.values()) == len(files)
    deeper = bl.summarise(bl.objects(f"file://{tmp_path}"), depth=2)
    assert deeper["volA/8_8_8"][:2] == [1, 100] and deeper["volA"] == [2, 15, True, True]


def test_main_prints_and_changes_nothing(tmp_path, capsys):
    _tree(tmp_path)
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    assert bl.main([f"file://{tmp_path}"]) == 0
    out = capsys.readouterr().out
    assert "volA" in out and "complete (info, provenance)" in out and "3 folders, 7 objects" in out
    assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*")) == before


def test_names_only_lists_the_top_level(tmp_path, capsys):
    _tree(tmp_path)
    assert bl.top_level(f"file://{tmp_path}") == ["loose.txt", "volA/", "volB/"]
    assert bl.main([f"file://{tmp_path}", "--names-only"]) == 0
    assert capsys.readouterr().out.split() == ["loose.txt", "volA/", "volB/"]
