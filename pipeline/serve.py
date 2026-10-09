"""HTTP server for viewing the rendered volume in neuroglancer.

Neuroglancer reads sharded Zarr v3 from the browser with HTTP range requests, so files are
served with CORS and single byte-range support (standard library only, no directory
listings). ``--bind 0.0.0.0`` lets anyone who can reach the node read the volume; it is
needed when tunnelling through the login host to a cluster node, but not with ``ssh -J``
to the node itself (both are printed on start).
"""

import logging
import os
import posixpath
import re
import socket
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from .cli import base_parser, setup
from .render import volume_name

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
            self.send_header("Content-Type", "application/json" if path.suffix == ".json" else "application/octet-stream")
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


def instructions(root, port, bind, name):
    """How to open the served volume in neuroglancer, locally and through an SSH tunnel."""
    node = socket.gethostname()
    user = os.environ.get("USER") or "<user>"
    login = os.environ.get("SLURM_SUBMIT_HOST") or "<login-host>"
    lines = [
        f"Serving {root} on http://{bind}:{port}/ (Ctrl-C to stop)",
        "",
        "In neuroglancer (e.g. https://neuroglancer-demo.appspot.com) add a layer with source",
        f"  zarr3://http://localhost:{port}/{name + '/' if name else ''}",
        "Chrome allows the https neuroglancer app to read from http://localhost",
        "(allow it if the browser asks for access to apps on this device or the local network).",
        "",
        "On a cluster node, first open a tunnel from your own machine:",
        f"  ssh -L {port}:{node}:{port} {user}@{login}",
    ]
    if bind in ("127.0.0.1", "localhost"):
        lines += [f"  (the login host connects to {node}:{port}, so start serve with --bind 0.0.0.0)",
                  f"or, if you may ssh to {node}, keep the default --bind (only you can then connect):",
                  f"  ssh -J {user}@{login} -L {port}:localhost:{port} {user}@{node}"]
    return "\n".join(lines)


def main(argv=None):
    p = base_parser(__doc__.splitlines()[0])
    p.add_argument("--port", type=int, help="HTTP port (default: serve.port, 8000; 0 picks a free port)")
    p.add_argument("--bind", help="listen address (default: serve.bind, 127.0.0.1)")
    p.add_argument("--dir", help="directory to serve (default: output_dir)")
    args = p.parse_args(argv)
    cfg = setup(args, DEFAULTS)
    root = Path(args.dir or cfg["output_dir"])
    if not root.is_dir():
        log.error("%s does not exist: run render first or pass --dir", root)
        return 1
    port = int(cfg["serve"]["port"] if args.port is None else args.port)
    bind = args.bind or cfg["serve"]["bind"]
    # Advertise the directory itself if it is the Zarr group, else the rendered volume in it.
    name = "" if (root / "zarr.json").exists() else volume_name(cfg)
    if name and not (root / name / "zarr.json").exists():
        log.warning("%s has no Zarr volume %s (yet): check render.name or pass --dir", root, name)
    try:
        server = Server(root, (bind, port))
    except OSError as e:  # typically the port is taken on a shared node
        log.error("cannot listen on %s:%d: %s (choose another --port)", bind, port, e)
        return 1
    print(instructions(root, server.server_address[1], bind, name), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
