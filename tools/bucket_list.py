"""What is in a Google Cloud Storage bucket, or under a prefix in it: the top-level folders with their object
counts and sizes, and whether each holds a Neuroglancer precomputed volume that looks complete (``info`` at its
top; ``provenance`` too, as em-pipe's export writes it last). Read-only: nothing is written or deleted.

    python tools/bucket_list.py gs://BUCKET[/PREFIX] [--credentials KEYFILE] [--depth 1]
    python tools/bucket_list.py file:///path/to/folder                      (a local folder, e.g. for tests)

--depth 2 also lists each folder's subfolders (e.g. a precomputed volume's scales). Credentials: a
service-account key file (default ~/.cloudvolume/secrets/google-secret.json, as upload.credentials), read by
the Google client only; the key is never copied, logged or printed.
"""
import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_KEY = "~/.cloudvolume/secrets/google-secret.json"


def objects(location, credentials=DEFAULT_KEY):
    """(name relative to the location, size in bytes) of every object under ``location``."""
    if location.startswith("file://"):
        root = Path(location.removeprefix("file://"))
        for dirpath, _, files in os.walk(root):
            for f in files:
                p = Path(dirpath) / f
                yield p.relative_to(root).as_posix(), p.stat().st_size
        return
    if not location.startswith("gs://"):
        sys.exit(f"bucket_list: {location}: give gs://BUCKET[/PREFIX] or file:///path")
    from google.cloud import storage
    from google.oauth2 import service_account
    bucket, _, prefix = location.removeprefix("gs://").partition("/")
    prefix = prefix.strip("/") + "/" if prefix.strip("/") else ""
    key = Path(credentials).expanduser()
    if not key.is_file():
        sys.exit(f"bucket_list: no key file at {key}")
    creds = service_account.Credentials.from_service_account_file(str(key))
    client = storage.Client(credentials=creds, project=creds.project_id)
    for b in client.list_blobs(bucket, prefix=prefix):
        yield b.name[len(prefix):], int(b.size or 0)


def summarise(items, depth=1):
    """{folder (the first ``depth`` path parts; '.' for objects at the top): [count, bytes, has info,
    has provenance]}; info / provenance count only at the folder's own top."""
    out = defaultdict(lambda: [0, 0, False, False])
    for name, size in items:
        parts = name.split("/")
        folder = "." if len(parts) == 1 else "/".join(parts[:min(depth, len(parts) - 1)])
        rec = out[folder]
        rec[0] += 1
        rec[1] += size
        rest = name[len(folder) + 1:] if folder != "." else name
        if rest == "info":
            rec[2] = True
        elif rest == "provenance":
            rec[3] = True
    return dict(sorted(out.items()))


def human(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1000 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1000


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("location", help="gs://BUCKET[/PREFIX] or file:///path")
    p.add_argument("--credentials", default=DEFAULT_KEY, help="service-account key file (never copied or printed)")
    p.add_argument("--depth", type=int, default=1, help="path parts that make a folder (default 1)")
    a = p.parse_args(argv)
    rows = summarise(objects(a.location, a.credentials), max(1, a.depth))
    if not rows:
        print(f"bucket_list: nothing under {a.location}")
        return 0
    width = max(len(k) for k in rows)
    print(f"{'folder':<{width}}  {'objects':>10}  {'size':>10}  volume")
    total_n = total_b = 0
    for k, (n, b, info, prov) in rows.items():
        kind = ("complete (info, provenance)" if info and prov else "info only" if info else "")
        print(f"{k:<{width}}  {n:>10,}  {human(b):>10}  {kind}")
        total_n += n
        total_b += b
    print(f"{len(rows)} folders, {total_n:,} objects, {human(total_b)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
