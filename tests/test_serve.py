"""Tests for pipeline.serve: byte ranges, HEAD, CORS preflight and errors over real HTTP."""

import http.client
import re
import socket
import threading

import numpy as np
import pytest

from pipeline import serve

DATA = np.random.default_rng(0).integers(0, 256, 1000, dtype=np.uint8).tobytes()
SHARD = "/volume.ome.zarr/s0/c/0"


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("serve")
    root = tmp_path / "render"
    (root / "volume.ome.zarr" / "s0" / "c").mkdir(parents=True)
    (root / "volume.ome.zarr" / "s0" / "c" / "0").write_bytes(DATA)
    (root / "volume.ome.zarr" / "zarr.json").write_text('{"zarr_format": 3}')
    (root / "empty.bin").write_bytes(b"")
    (root / "with space.bin").write_bytes(b"abc")
    (tmp_path / "secret.txt").write_text("outside the served directory")
    srv = serve.Server(root, ("127.0.0.1", 0))
    thread = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    yield srv.server_address[1]
    srv.shutdown()
    srv.server_close()


def request(port, method, path, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request(method, path, headers=headers or {})
    resp = conn.getresponse()
    body = resp.read()
    conn.close()
    return resp.status, {k.lower(): v for k, v in resp.getheaders()}, body


def test_get_whole_file(server):
    status, headers, body = request(server, "GET", SHARD)
    assert status == 200 and body == DATA
    assert headers["content-length"] == "1000" and headers["accept-ranges"] == "bytes"
    assert headers["access-control-allow-origin"] == "*"
    assert "content-range" not in headers
    status, headers, body = request(server, "GET", "/volume.ome.zarr/zarr.json?x=1")
    assert status == 200 and body == b'{"zarr_format": 3}' and headers["content-type"] == "application/json"
    assert request(server, "GET", "/with%20space.bin")[2] == b"abc"


@pytest.mark.parametrize("rng, start, stop", [
    ("bytes=0-9", 0, 10),          # first bytes
    ("bytes=100-199", 100, 200),   # middle
    ("bytes=990-", 990, 1000),     # open-ended
    ("bytes=995-5000", 995, 1000), # end clipped to the file
    ("bytes=-16", 984, 1000),      # suffix
    ("bytes=-5000", 0, 1000),      # suffix longer than the file
    ("bytes=999-999", 999, 1000),  # last byte
    ("Bytes=0-9", 0, 10),          # range units are case-insensitive
])
def test_range(server, rng, start, stop):
    status, headers, body = request(server, "GET", SHARD, {"Range": rng})
    assert status == 206
    assert body == DATA[start:stop]
    assert headers["content-range"] == f"bytes {start}-{stop - 1}/1000"
    assert headers["content-length"] == str(stop - start)
    assert headers["access-control-allow-origin"] == "*"
    assert "content-range" in headers["access-control-expose-headers"].lower()


@pytest.mark.parametrize("path, rng", [(SHARD, "bytes=1000-"), (SHARD, "bytes=5000-6000"),
                                       (SHARD, "bytes=-0"), ("/empty.bin", "bytes=-10")])
def test_unsatisfiable_range(server, path, rng):
    status, headers, body = request(server, "GET", path, {"Range": rng})
    size = 1000 if path == SHARD else 0
    assert status == 416 and body == b""
    assert headers["content-range"] == f"bytes */{size}"
    assert headers["access-control-allow-origin"] == "*"


@pytest.mark.parametrize("rng", ["bytes=0-1,5-9", "bytes=9-2", "items=0-9", "bytes=-"])
def test_ignored_range_sends_whole_file(server, rng):
    status, _, body = request(server, "GET", SHARD, {"Range": rng})
    assert status == 200 and body == DATA


def test_head(server):
    status, headers, body = request(server, "HEAD", SHARD)
    assert status == 200 and body == b""
    assert headers["content-length"] == "1000" and headers["accept-ranges"] == "bytes"
    status, headers, _ = request(server, "HEAD", SHARD, {"Range": "bytes=10-19"})
    assert status == 206 and headers["content-range"] == "bytes 10-19/1000"


def test_options_preflight(server):
    status, headers, body = request(server, "OPTIONS", SHARD, {
        "Origin": "https://neuroglancer-demo.appspot.com", "Access-Control-Request-Method": "GET",
        "Access-Control-Request-Headers": "range"})
    assert status == 204 and body == b""
    assert headers["access-control-allow-origin"] == "*"
    assert "range" in headers["access-control-allow-headers"].lower()
    assert {"GET", "HEAD"} <= {m.strip() for m in headers["access-control-allow-methods"].split(",")}
    expose = {h.strip().lower() for h in headers["access-control-expose-headers"].split(",")}
    assert {"content-range", "content-length", "accept-ranges"} <= expose
    _, headers, _ = request(server, "OPTIONS", "/anything")
    assert headers["access-control-allow-headers"] == "Range"


@pytest.mark.parametrize("path", ["/missing", "/volume.ome.zarr/s0/c/1", "/", "/volume.ome.zarr",
                                  "/../secret.txt", "/%2e%2e/secret.txt", "/volume.ome.zarr/../../secret.txt",
                                  "/volume.ome.zarr/%00"])
def test_not_found(server, path):
    status, headers, body = request(server, "GET", path)
    assert status == 404 and body == b""
    assert headers["access-control-allow-origin"] == "*"
    assert request(server, "HEAD", path)[0] == 404


def test_keep_alive_serves_several_requests(server):
    conn = http.client.HTTPConnection("127.0.0.1", server, timeout=10)
    for start in (0, 500, 900):
        conn.request("GET", SHARD, headers={"Range": f"bytes={start}-{start + 49}"})
        resp = conn.getresponse()
        assert resp.status == 206 and resp.read() == DATA[start:start + 50]
    conn.request("GET", "/missing")
    resp = conn.getresponse()
    assert resp.status == 404 and resp.read() == b""
    conn.request("GET", SHARD)
    assert conn.getresponse().read() == DATA
    conn.close()


def test_instructions(monkeypatch):
    monkeypatch.setenv("USER", "alice")
    monkeypatch.setenv("SLURM_SUBMIT_HOST", "login01")
    monkeypatch.setattr(serve.socket, "gethostname", lambda: "ca123")
    text = serve.instructions("/out/render", 8123, "127.0.0.1", "volume.ome.zarr")
    assert "zarr3://http://localhost:8123/volume.ome.zarr/" in text
    assert "ssh -L 8123:ca123:8123 alice@login01" in text
    assert "--bind 0.0.0.0" in text
    assert "ssh -J alice@login01 -L 8123:localhost:8123 alice@ca123" in text
    text = serve.instructions("/out/render/volume.ome.zarr", 8000, "0.0.0.0", "")
    assert "zarr3://http://localhost:8000/\n" in text and "--bind 0.0.0.0" not in text
    assert "ssh -J" not in text


def test_main_rejects_missing_dir(make_config, tmp_path):
    cfg = make_config(tmp_path / "raw")
    assert serve.main(["--config", str(cfg), "--port", "0"]) == 1


def test_main_reports_port_in_use(make_config, tmp_path, caplog):
    cfg = make_config(tmp_path / "raw")
    (tmp_path / "out" / "render").mkdir(parents=True)
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        port = taken.getsockname()[1]
        assert serve.main(["--config", str(cfg), "--port", str(port)]) == 1
    assert "choose another --port" in caplog.text


def test_main_prints_viewing_instructions(make_config, tmp_path, capsys, monkeypatch):
    cfg = make_config(tmp_path / "raw", render={"name": "v.ome.zarr"})
    (tmp_path / "out" / "render").mkdir(parents=True)

    def interrupt(self):
        raise KeyboardInterrupt

    monkeypatch.setattr(serve.Server, "serve_forever", interrupt)
    assert serve.main(["--config", str(cfg), "--port", "0"]) == 0
    out = capsys.readouterr().out
    port = re.search(r"zarr3://http://localhost:(\d+)/v\.ome\.zarr/", out).group(1)
    assert int(port) > 0 and str(tmp_path / "out" / "render") in out


def test_tensorstore_reads_sharded_volume_over_http(tmp_path):
    """A sharded Zarr v3 client (range reads of shard index and chunks) sees the written data."""
    import tensorstore as ts
    from pipeline import omezarr
    root = tmp_path / "v.ome.zarr"
    omezarr.create(root, (20, 40, 36), (8.0, 8.0, 8.0), num_scales=2, chunk=(4, 8, 8), shard=(8, 16, 16))
    data = np.random.default_rng(1).integers(0, 256, (20, 40, 36), dtype=np.uint8)
    omezarr.open_scale(root, 0).write(data).result()
    srv = serve.Server(tmp_path, ("127.0.0.1", 0))
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}/v.ome.zarr"
        s0 = ts.open({"driver": "zarr3", "kvstore": f"{url}/s0/"}, open=True, read=True).result()
        np.testing.assert_array_equal(s0[3:9, 5:30, 2:20].read().result(), data[3:9, 5:30, 2:20])
        np.testing.assert_array_equal(s0.read().result(), data)
        s1 = ts.open({"driver": "zarr3", "kvstore": f"{url}/s1/"}, open=True, read=True).result()
        assert not s1.read().result().any()   # unwritten shards are 404 -> fill value
    finally:
        srv.shutdown()
        srv.server_close()
