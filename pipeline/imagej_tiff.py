"""Reader for the raw ImageJ TIFF stacks.

The acquisition exports are ImageJ stacks larger than 4 GB written as *classic* (not Big)
TIFF: a single IFD, then every slice stored back to back, uncompressed, big-endian.
Generic TIFF readers that walk IFDs see only one page. Here we read the first IFD and
the ImageJ description (``images=N``), then memory-map the pixel data, so reading
slice ``i`` (or a block of rows/columns of it) only touches those bytes.

Ordinary multi-page TIFFs (e.g. small test files) also work: if the slices are
contiguous they are memory-mapped the same way, otherwise pages are read with tifffile.
"""

import os
from functools import cached_property

import numpy as np
import tifffile

_UNIT_NM = {"nm": 1.0, "nanometer": 1.0, "micron": 1000.0, "um": 1000.0, "µm": 1000.0,
            "\\u00B5m": 1000.0, "mm": 1e6}


class ImageJStack:
    """Lazy view of one TIFF stack. Header is read on construction; pixels on demand."""

    def __init__(self, path):
        self.path = str(path)
        self.size_bytes = os.path.getsize(self.path)
        with tifffile.TiffFile(self.path) as tif:
            page = tif.pages.first
            self.byteorder = tif.byteorder
            self.is_bigtiff = tif.is_bigtiff
            self.imagej_metadata = tif.imagej_metadata or {}
            if page.samplesperpixel != 1:
                raise ValueError(f"{self.path}: expected single-channel images")
            self.height, self.width = int(page.imagelength), int(page.imagewidth)
            self.dtype = np.dtype(page.dtype).newbyteorder(self.byteorder)
            self.compressed = page.compression != 1
            self.data_offset = int(page.dataoffsets[0]) if len(page.dataoffsets) else None
            single_segment = len(page.dataoffsets) == 1
            self.n = int(self.imagej_metadata.get("images") or 0) or len(tif.pages)
            # Contiguous if slice 1 starts right after slice 0. With a single IFD (ImageJ >4 GB)
            # there is no second page to check, and ImageJ's format guarantees contiguity.
            if self.compressed or not single_segment:
                self.contiguous = False
            elif self.n == 1 or len(tif.pages) == 1:
                self.contiguous = True
            else:
                second = tif.pages[1]
                self.contiguous = (len(second.dataoffsets) == 1
                                   and int(second.dataoffsets[0]) == self.data_offset + self.slice_bytes)
            self._xres = page.tags["XResolution"].value if "XResolution" in page.tags else None
            self._yres = page.tags["YResolution"].value if "YResolution" in page.tags else None

    # ----- metadata -----------------------------------------------------------------

    @property
    def shape(self):
        return (self.n, self.height, self.width)

    @property
    def slice_bytes(self):
        return self.height * self.width * self.dtype.itemsize

    @property
    def expected_size(self):
        """Smallest file size that holds every slice (contiguous stacks only)."""
        return self.data_offset + self.n * self.slice_bytes

    @property
    def truncated(self):
        return self.contiguous and self.size_bytes < self.expected_size

    @property
    def labels(self):
        labels = self.imagej_metadata.get("Labels") or []
        # tifffile gives a single-slice stack's label as a str, not a list of one.
        return [labels] if isinstance(labels, str) else list(labels)

    @property
    def voxel_size_nm(self):
        """(x, y, z) in nm, or None for any axis the file doesn't record."""
        unit = str(self.imagej_metadata.get("unit", "")).strip()
        scale = _UNIT_NM.get(unit)

        def from_res(res):
            if res is None or scale is None:
                return None
            num, den = res
            return scale * den / num if num else None

        z = self.imagej_metadata.get("spacing")
        return (from_res(self._xres), from_res(self._yres),
                scale * float(z) if (z is not None and scale is not None) else None)

    # ----- pixels -------------------------------------------------------------------

    @cached_property
    def _memmap(self):
        if not self.contiguous:
            raise ValueError(f"{self.path}: slices are not stored contiguously")
        n = self.n
        if self.truncated:  # map only complete slices
            n = max(0, (self.size_bytes - self.data_offset) // self.slice_bytes)
        return np.memmap(self.path, dtype=self.dtype, mode="r", offset=self.data_offset,
                         shape=(n, self.height, self.width))

    def memmap(self):
        """Read-only memmap of shape (n, height, width) in the file's byte order."""
        return self._memmap

    def read(self, i, rows=slice(None), cols=slice(None)):
        """Slice ``i`` (optionally a block of it) as a native-endian array copy.

        Contiguous stacks are read with plain file reads into a fresh buffer rather than through
        the memmap: pages read through a mapping stay counted in the process's resident memory,
        so a task reading many 354 MB slices would look like it uses tens of GB.
        """
        if not 0 <= i < self.n:
            raise IndexError(f"{self.path}: slice {i} out of range 0..{self.n - 1}")
        native = self.dtype.newbyteorder("=")
        if not self.contiguous:
            with tifffile.TiffFile(self.path) as tif:
                block = tif.pages[i].asarray()[rows, cols]
            return np.ascontiguousarray(block).astype(native, copy=False)
        if not (isinstance(rows, slice) and isinstance(cols, slice)):
            return self.read(i)[rows, cols]
        r0, r1, rs = rows.indices(self.height)
        c0, c1, cs = cols.indices(self.width)
        if rs < 0 or cs < 0:   # reversed steps: not used by the pipeline, keep it simple
            return self.read(i)[rows, cols]
        nr, nc = max(0, r1 - r0), max(0, c1 - c0)
        item, w = self.dtype.itemsize, self.width
        base = self.data_offset + i * self.slice_bytes
        with open(self.path, "rb", buffering=0) as fh:
            if nr == 0 or nc == 0:
                out = np.empty((nr, nc), self.dtype)
            elif 4 * nc >= w:   # wide block: whole rows in one read, then crop
                out = np.empty((nr, w), self.dtype)
                _read_into(fh, base + r0 * w * item, memoryview(out).cast("B"), self.path)
                out = out[:, c0:c1]
            else:               # narrow strip: one read per row
                out = np.empty((nr, nc), self.dtype)
                buf, step = memoryview(out).cast("B"), nc * item
                for k in range(nr):
                    _read_into(fh, base + ((r0 + k) * w + c0) * item, buf[k * step:(k + 1) * step], self.path)
        out = out[::rs, ::cs]
        return np.ascontiguousarray(out).astype(native, copy=False) if out.dtype != native else np.ascontiguousarray(out)

    def read_raw_bytes(self, offset, length):
        """Raw bytes from the pixel data section (for cheap content hashing)."""
        with open(self.path, "rb") as fh:
            fh.seek(self.data_offset + offset)
            return fh.read(length)


def _read_into(fh, offset, buf, path):
    """Fill ``buf`` from ``offset``; network filesystems may return short reads."""
    fh.seek(offset)
    done = 0
    while done < len(buf):
        n = fh.readinto(buf[done:])
        if not n:
            raise EOFError(f"{path}: file ends before byte {offset + len(buf)} (truncated?)")
        done += n
