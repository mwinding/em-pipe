"""tools/view_link.py: the link holds the layers (first one visible), sources, dimensions, position and layout
asked for, and neuroglancer (when installed) reads it back the same."""

import json
import sys
from pathlib import Path
from urllib.parse import unquote

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import view_link  # noqa: E402

VOXEL = (7.999856450575851e-09, 7.999856450575851e-09, 7.999856447888447e-09)


def _decode(url):
    assert url.startswith(view_link.PREFIX)
    return json.loads(unquote(url[len(view_link.PREFIX):]))


def test_link_state():
    layers = [("plain (old)", "a.ome.zarr"), ("plain, new", "b.ome.zarr"), ("pre", "c.precomputed")]
    s = view_link.state(layers, 8001, VOXEL, (10.5, 20.5, 3.5), "4panel")
    d = _decode(view_link.link(s))
    assert d == s
    assert [lay["name"] for lay in d["layers"]] == ["plain (old)", "plain, new", "pre"]
    assert [lay["visible"] for lay in d["layers"]] == [True, False, False]
    assert d["layers"][0]["source"] == "zarr3://http://localhost:8001/a.ome.zarr/"
    assert d["layers"][2]["source"] == "precomputed://http://localhost:8001/c.precomputed/"
    assert d["dimensions"] == {"x": [VOXEL[0], "m"], "y": [VOXEL[1], "m"], "z": [VOXEL[2], "m"]}
    assert d["position"] == [10.5, 20.5, 3.5] and d["layout"] == "4panel"


def test_main_writes_file(tmp_path, capsys):
    out = tmp_path / "view.txt"
    view_link.main(["--port", "8001", "--position", "1", "2", "3", "--layout", "4panel", "--out", str(out),
                    "old=a.ome.zarr/", "new=b.ome.zarr"])
    url = capsys.readouterr().out.strip()
    assert out.read_text().strip() == url
    d = _decode(url)
    assert [lay["source"] for lay in d["layers"]] == ["zarr3://http://localhost:8001/a.ome.zarr/",
                                                      "zarr3://http://localhost:8001/b.ome.zarr/"]
    with pytest.raises(SystemExit):
        view_link.main(["no-volume"])


def test_neuroglancer_reads_it_back():
    pytest.importorskip("neuroglancer")
    s = view_link.state([("old", "a.ome.zarr"), ("new", "b.ome.zarr")], 8001, VOXEL, (1.5, 2.5, 3.5), "4panel")
    assert view_link.check(view_link.link(s), s) == []
    other = dict(s, layout="xy")
    assert view_link.check(view_link.link(s), other)
    zoomed = view_link.state([("old", "a.ome.zarr")], 8001, VOXEL, (1.5, 2.5, 3.5), "4panel", cross_section_scale=2)
    assert zoomed["crossSectionScale"] == 2.0 and view_link.check(view_link.link(zoomed), zoomed) == []
    assert view_link.check(view_link.link(zoomed), dict(zoomed, crossSectionScale=1.0))
