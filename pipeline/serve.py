"""HTTP server for viewing rendered volumes in neuroglancer.

    python3 -m pipeline.serve FOLDER          # any Python 3, from the repo root; no config needed
    python -m pipeline.serve --config C      # serves the config's output_dir

Prints a neuroglancer link with every volume in the folder (``*.zarr`` with a zarr.json, and
Neuroglancer precomputed ``*.precomputed`` with an info file) as a layer. Neuroglancer reads sharded
Zarr v3 and sharded precomputed from the browser with HTTP range requests, so files
are served with CORS and single byte-range support (standard library only, no directory
listings). ``--bind 0.0.0.0`` lets anyone who can reach the node read the volume; it is
needed when tunnelling through the login host to a cluster node, but not with ``ssh -J``
to the node itself (both are printed on start).
"""

import argparse
import json
import logging
import os
import posixpath
import re
import socket
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

log = logging.getLogger(__name__)

DEFAULTS = {
    "serve": {
        "port": 8000,           # HTTP port (--port overrides)
        "bind": "127.0.0.1",    # listen address (--bind overrides); 0.0.0.0 to tunnel via the login host
    },
}


def parse_range(header, size):
    """(start, stop) of a single ``bytes=`` range, or None to send the whole file.

    Raises ValueError if the range cannot be satisfied (416). Malformed and multi-range
    headers are ignored, which RFC 9110 allows.
    """
    m = re.fullmatch(r"\s*bytes\s*=\s*(\d*)\s*-\s*(\d*)\s*", header or "", re.IGNORECASE)
    if not m or m.groups() == ("", ""):
        return None
    first, last = m.groups()
    if not first:  # suffix range: the last N bytes
        if int(last) == 0 or size == 0:
            raise ValueError("empty suffix range")
        return max(0, size - int(last)), size
    start = int(first)
    if last and int(last) < start:
        return None
    if start >= size:
        raise ValueError("range starts beyond the end of the file")
    return start, min(int(last) + 1, size) if last else size


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # keep-alive: neuroglancer makes many small requests

    def end_headers(self):
        # On every response, errors included: neuroglancer must see 404s (missing chunks) through CORS.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Expose-Headers", "Content-Range, Content-Length, Accept-Ranges")
        self.send_header("Cache-Control", "no-cache")  # the volume may be (re)written while viewed
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", self.headers.get("Access-Control-Request-Headers") or "Range")
        # Chrome's preflight before a public https page may read from localhost or a private network.
        self.send_header("Access-Control-Allow-Private-Network", "true")
        self.send_header("Access-Control-Max-Age", "86400")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        self._send_file(body=True)

    def do_HEAD(self):
        self._send_file(body=False)

    def _send_file(self, body):
        # normpath of an absolute path drops any ".." above the root.
        rel = posixpath.normpath("/" + unquote(urlsplit(self.path).path)).lstrip("/")
        path = self.server.root / rel
        try:
            fh = open(path, "rb")
        except (OSError, ValueError):  # missing, a directory, or an invalid name
            return self._empty(404)
        with fh:
            size = os.fstat(fh.fileno()).st_size
            try:
                rng = parse_range(self.headers.get("Range"), size)
            except ValueError:
                return self._empty(416, {"Content-Range": f"bytes */{size}"})
            start, stop = rng or (0, size)
            self.send_response(206 if rng else 200)
            is_json = path.suffix == ".json" or path.name in ("info", "provenance")
            self.send_header("Content-Type", "application/json" if is_json else "application/octet-stream")
            self.send_header("Content-Length", str(stop - start))
            self.send_header("Accept-Ranges", "bytes")
            if rng:
                self.send_header("Content-Range", f"bytes {start}-{stop - 1}/{size}")
            self.end_headers()
            if body and stop > start:
                self.connection.sendfile(fh, start, stop - start)

    def _empty(self, code, headers=None):
        self.send_response(code)
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, format, *args):
        log.debug("%s %s", self.address_string(), format % args)


class Server(ThreadingHTTPServer):
    """Threading HTTP server for the files under ``root``."""

    def __init__(self, root, address):
        self.root = Path(root)
        super().__init__(address, Handler)

    def handle_error(self, request, client_address):
        # Neuroglancer cancels requests whenever the view moves; a dropped connection is not an error.
        if not isinstance(sys.exc_info()[1], ConnectionError):
            super().handle_error(request, client_address)


def _precomputed(path):
    """Is ``path`` a Neuroglancer precomputed volume (a folder with a multiscale volume info)?"""
    try:
        return json.loads((Path(path) / "info").read_text()).get("@type") == "neuroglancer_multiscale_volume"
    except (OSError, ValueError, AttributeError):
        return False


def volumes(root):
    """Volumes to show: the folder itself if it is one, else its ``*.zarr`` and ``*.precomputed``
    subfolders."""
    root = Path(root)
    if (root / "zarr.json").exists() or _precomputed(root):
        return [""]
    return sorted(d.name for d in root.iterdir()
                  if (d.name.endswith(".zarr") and (d / "zarr.json").exists())
                  or (d.name.endswith(".precomputed") and _precomputed(d)))


def voxel_nm(volume):
    """{axis: size in nm} of a volume's finest scale (OME metadata or precomputed info), or None."""
    try:
        if _precomputed(volume):
            res = json.loads((Path(volume) / "info").read_text())["scales"][0]["resolution"]
            return dict(zip("xyz", map(float, res)))
        ms = json.loads((Path(volume) / "zarr.json").read_text())["attributes"]["ome"]["multiscales"][0]
        scale = ms["datasets"][0]["coordinateTransformations"][0]["scale"]
        return {a["name"]: float(v) for a, v in zip(ms["axes"], scale)}
    except (OSError, KeyError, IndexError, TypeError, ValueError):
        return None


def source(port, name, root=None):
    """Neuroglancer data source URL of a served volume."""
    path = f"http://localhost:{port}/{name + '/' if name else ''}"
    kind = "precomputed" if (root is not None and _precomputed(Path(root) / name)) or name.endswith(".precomputed") else "zarr3"
    return f"{kind}://{path}"


def neuroglancer_link(port, names, root=None):
    """neuroglancer-demo link with one image layer per volume (the first one visible), opening on
    the x-y imaging plane.

    The volumes' axes are (z, y, x) and neuroglancer's xy panel shows the first two global
    dimensions, so the link declares them in x, y, z order (at the finest voxel size of the
    volumes); otherwise the xy panel is a z-y side view.
    """
    layers = [{"type": "image", "source": source(port, name, root),
               "name": name.removesuffix(".precomputed").removesuffix(".zarr").removesuffix(".ome") or "volume",
               "visible": i == 0}
              for i, name in enumerate(names)]
    sizes = [v for v in (voxel_nm(Path(root) / n) for n in names) if v] if root is not None else []
    dims = {a: [min((v.get(a, 8.0) for v in sizes), default=8.0) * 1e-9, "m"] for a in ("x", "y", "z")}
    state = {"dimensions": dims, "displayDimensions": ["x", "y", "z"], "layers": layers, "layout": "xy"}
    return "https://neuroglancer-demo.appspot.com/#!" + quote(json.dumps(state, separators=(",", ":")), safe=":/,")


def instructions(root, port, bind, names):
    """How to open the served volumes in neuroglancer, locally and through an SSH tunnel."""
    node = socket.gethostname()
    user = os.environ.get("USER") or "<user>"
    login = os.environ.get("SLURM_SUBMIT_HOST") or "<login-host>"
    lines = [f"Serving {root} on http://{bind}:{port}/ (Ctrl-C to stop)", ""]
    if names:
        lines += ["Open in Chrome (layers: " + ", ".join(n or "volume" for n in names) + "):",
                  "  " + neuroglancer_link(port, names, root), "",
                  "or add a layer by hand in https://neuroglancer-demo.appspot.com with source:"]
        lines += [f"  {source(port, n, root)}" for n in names]
    else:
        lines += [f"No volumes (*.zarr with a zarr.json, *.precomputed with an info) found in {root}."]
    lines += ["Chrome allows the https neuroglancer app to read from http://localhost",
              "(allow it if the browser asks for access to apps on this device or the local network).",
              "",
              "On a cluster node, first open a tunnel from your own machine:",
              f"  ssh -L {port}:{node}:{port} {user}@{login}"]
    if bind in ("127.0.0.1", "localhost"):
        lines += [f"  (the login host connects to {node}:{port}, so start serve with --bind 0.0.0.0)",
                  f"or, if you may ssh to {node}, keep the default --bind (only you can then connect):",
                  f"  ssh -J {user}@{login} -L {port}:localhost:{port} {user}@{node}"]
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", nargs="?", help="folder with rendered volumes (or a volume itself)")
    p.add_argument("--config", help="dataset config: serve its output_dir")
    p.add_argument("--output-dir", help="with --config: override output_dir")
    p.add_argument("--dir", help=argparse.SUPPRESS)   # old name for FOLDER
    p.add_argument("--port", type=int, help="HTTP port (default: serve.port, 8000; 0 picks a free port)")
    p.add_argument("--bind", help="listen address (default: serve.bind, 127.0.0.1)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    opts = dict(DEFAULTS["serve"])
    folder = args.folder or args.dir
    if folder is None:
        if not args.config:
            p.error("give a FOLDER to serve, or --config")
        from .config import load_config   # needs PyYAML: only in config mode
        cfg = load_config(args.config, output_dir=args.output_dir, step_defaults=DEFAULTS)
        opts.update(cfg["serve"])
        folder = cfg["output_dir"]
    root = Path(folder)
    if not root.is_dir():
        log.error("%s does not exist", root)
        return 1
    port = int(opts["port"] if args.port is None else args.port)
    bind = args.bind or opts["bind"]
    try:
        server = Server(root, (bind, port))
    except OSError as e:  # typically the port is taken on a shared node
        log.error("cannot listen on %s:%d: %s (choose another --port)", bind, port, e)
        return 1
    print(instructions(root, server.server_address[1], bind, volumes(root)), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
