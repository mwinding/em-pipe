"""Upload a finished output folder (by default the precomputed export) to a Google Cloud Storage bucket.

``run`` (array): every task takes its share of the folder's files (sorted, every n-th), lists what
the bucket already has under the destination, and uploads what is missing or of another size, with
CRC32C checked by the client. An object only appears in GCS once complete, so a re-run continues
where one stopped. The top-level ``info`` and ``provenance`` files wait for ``finish``: it checks that
every other file is in the bucket, uploads them last (so the volume only looks complete when it is),
then checks everything once more. ``--dry-run`` only counts.

Credentials: a service-account key file at ``upload.credentials`` (default
~/.cloudvolume/secrets/google-secret.json, where CloudVolume looks), read by the Google client only;
the key is never copied, logged or printed. Keep it outside the repository (``chmod 600``); a key
inside the repository is refused. ``upload.destination`` may also be ``file:///path`` (a local copy,
for tests).
"""

import logging
import os
import shutil
import stat
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .cli import base_parser, setup, task_info
from .precomputed import DEFAULTS as PRECOMPUTED_DEFAULTS, paths as precomputed_paths
from .render import DEFAULTS as RENDER_DEFAULTS

log = logging.getLogger(__name__)

DEFAULTS = {
    "upload": {
        "enabled": False,     # run_pipeline.sh: upload after the precomputed export
        "source": None,       # folder in output_dir to upload; null: the precomputed export
        "destination": None,  # gs://bucket/prefix (or file:///path)
        "credentials": "~/.cloudvolume/secrets/google-secret.json",   # service-account key file
        "threads": 16,        # parallel uploads per task
        "retries": 4,
    }
}
LAST = ("info", "provenance")   # top-level files uploaded by finish, after everything else
REPO = Path(__file__).resolve().parent.parent


def source_dir(cfg):
    u = cfg["upload"]
    if u["source"]:
        return Path(cfg["output_dir"]) / u["source"]
    return precomputed_paths(cfg)[1]


def local_files(root):
    """{relative posix path: size} of every file under root."""
    root = Path(root)
    return {p.relative_to(root).as_posix(): p.stat().st_size for p in sorted(root.rglob("*")) if p.is_file()}


class LocalTarget:
    """A folder standing in for a bucket (file:///path destinations, tests)."""

    def __init__(self, path):
        self.root = Path(path)

    def list(self):
        return local_files(self.root) if self.root.exists() else {}

    def put(self, local, rel):
        dest = self.root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        shutil.copyfile(local, tmp)
        os.replace(tmp, dest)


class GCSTarget:
    """A bucket prefix, with the service-account key at ``credentials``; ``threads`` uploads share
    the client, so its connection pool holds as many connections (requests' default is 10: more
    threads would reconnect for every file)."""

    def __init__(self, destination, credentials, threads=16):
        from google.cloud import storage
        from google.oauth2 import service_account
        from requests.adapters import HTTPAdapter
        bucket, _, prefix = destination.removeprefix("gs://").partition("/")
        self.prefix = prefix.strip("/") + "/" if prefix.strip("/") else ""
        creds = service_account.Credentials.from_service_account_file(str(credentials))
        client = storage.Client(credentials=creds, project=creds.project_id)
        n = max(10, int(threads))
        client._http.mount("https://", HTTPAdapter(pool_connections=n, pool_maxsize=n))
        self.bucket = client.bucket(bucket)

    def list(self):
        n = len(self.prefix)
        return {b.name[n:]: b.size for b in self.bucket.list_blobs(prefix=self.prefix)}

    def put(self, local, rel):
        blob = self.bucket.blob(self.prefix + rel)
        kind = "application/json" if Path(rel).name in LAST or rel.endswith(".json") else "application/octet-stream"
        blob.upload_from_filename(str(local), content_type=kind, checksum="crc32c")


def credentials_path(u):
    """The key file's path, after checks that it exists and lies outside the repository."""
    path = Path(os.path.expanduser(str(u["credentials"]))).resolve()
    if REPO in path.parents:
        raise ValueError(f"upload.credentials {path} is inside the repository: keep the key elsewhere")
    if not path.is_file():
        raise FileNotFoundError(f"upload.credentials: no key file at {path}")
    if path.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        log.warning("%s is readable by others: chmod 600 it", path)
    return path


def target(cfg):
    u = cfg["upload"]
    dest = u["destination"]
    if not dest:
        raise ValueError("upload.destination is not set (gs://bucket/prefix)")
    if dest.startswith("file://"):
        return LocalTarget(dest.removeprefix("file://"))
    if dest.startswith("gs://"):
        return GCSTarget(dest, credentials_path(u), u["threads"])
    raise ValueError(f"upload.destination must start with gs:// or file://, not {dest!r}")


def _put_all(tgt, root, todo, threads, retries):
    """Upload ``todo`` [(rel, size)] with retries; returns (files, bytes) done, raises if any fails."""
    done = sent = 0
    t0 = time.time()

    def one(rel):
        for attempt in range(retries + 1):
            try:
                tgt.put(Path(root) / rel, rel)
                return
            except Exception as e:   # network errors: retry with backoff
                if attempt == retries:
                    raise RuntimeError(f"{rel}: {e}") from e
                time.sleep(2 ** attempt)

    size = dict(todo)
    with ThreadPoolExecutor(max(1, int(threads))) as pool:
        futures = {pool.submit(one, rel): rel for rel, _ in todo}
        for f in as_completed(futures):
            f.result()
            done += 1
            sent += size[futures[f]]
            if done % 200 == 0 or done == len(todo):
                log.info("%d/%d files, %.1f GB, %.0f MB/s", done, len(todo), sent / 1e9,
                         sent / 1e6 / max(time.time() - t0, 1e-3))
    return done, sent


def run(cfg, task_id, num_tasks, dry_run=False):
    u = cfg["upload"]
    root = source_dir(cfg)
    if not root.is_dir():
        log.error("%s not found: nothing to upload", root)
        return 1
    files = local_files(root)
    names = [r for r in files if r not in LAST]
    mine = names[task_id::num_tasks]
    tgt = target(cfg)
    have = tgt.list()
    todo = [(r, files[r]) for r in mine if have.get(r) != files[r]]
    log.info("task %d/%d: %d of %d files, %d to upload (%.1f GB) to %s", task_id, num_tasks, len(mine),
             len(names), len(todo), sum(s for _, s in todo) / 1e9, u["destination"])
    if dry_run or not todo:
        return 0
    _put_all(tgt, root, todo, u["threads"], u["retries"])
    return 0


def finish(cfg, dry_run=False):
    """Check every data file is in the bucket, then upload info/provenance, then check all."""
    u = cfg["upload"]
    root = source_dir(cfg)
    files = local_files(root)
    tgt = target(cfg)
    have = tgt.list()
    bad = [r for r, s in files.items() if r not in LAST and have.get(r) != s]
    if bad:
        log.error("%d files missing or of another size in %s, e.g. %s: run 'upload run' (all tasks) first",
                  len(bad), u["destination"], bad[:5])
        return 1
    last = [(r, files[r]) for r in LAST if r in files]
    if dry_run:
        log.info("all %d data files uploaded; would upload %s", len(files) - len(last), [r for r, _ in last])
        return 0
    _put_all(tgt, root, last, 1, u["retries"])
    have = tgt.list()
    bad = [r for r, s in files.items() if have.get(r) != s]
    if bad:
        log.error("after upload, %d files differ, e.g. %s", len(bad), bad[:5])
        return 1
    log.info("%s: all %d files (%.1f GB) in %s", root.name, len(files), sum(files.values()) / 1e9, u["destination"])
    return 0


def main(argv=None):
    p = base_parser(__doc__.splitlines()[0])
    p.add_argument("command", choices=["run", "finish"])
    p.add_argument("--dry-run", action="store_true", help="count what would be uploaded, upload nothing")
    args = p.parse_args(argv)
    cfg = setup(args, {**RENDER_DEFAULTS, **PRECOMPUTED_DEFAULTS, **DEFAULTS})
    if args.command == "run":
        return run(cfg, *task_info(args), dry_run=args.dry_run)
    return finish(cfg, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
