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
        """Slice ``i`` (optionally a block of it) as a native-endian array copy."""
        if not 0 <= i < self.n:
            raise IndexError(f"{self.path}: slice {i} out of range 0..{self.n - 1}")
        if self.contiguous:
            block = self._memmap[i, rows, cols]
        else:
            with tifffile.TiffFile(self.path) as tif:
                block = tif.pages[i].asarray()[rows, cols]
        return np.ascontiguousarray(block).astype(self.dtype.newbyteorder("="), copy=False)

    def read_raw_bytes(self, offset, length):
        """Raw bytes from the pixel data section (for cheap content hashing)."""
        with open(self.path, "rb") as fh:
            fh.seek(self.data_offset + offset)
            return fh.read(length)
