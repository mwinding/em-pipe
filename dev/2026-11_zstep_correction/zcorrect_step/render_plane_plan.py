"""The part of pipeline/render.py that used the zcorrect step's positions, as it was when the step was
removed from the pipeline (26 Nov 2026). A record for reviving the step: nothing imports this file.

In pipeline/render.py these sat between ``to_volume`` and ``_destreak_settings``. The plane plan is a list
with one entry per output plane, ``[(z, weight), ...]``: the source slices the plane is a weighted mean of.
Without positions a plane is the mean of f consecutive selected slices (this is all render does now); with
positions, planes are placed ``step_nm`` apart and each interpolates linearly between the two slices either
side of it.

How render used them (the hooks to put back):

- ``init``::

      vz, vy, vx = voxel_size_nm(cfg, plan["file"].unique())
      positions = _positions(cfg, zs)
      planes = plane_sources(zs, f, positions, vz * f)
      meta = {..., "zcorrected": positions is not None, ..., "planes": planes}

  ``planes`` and ``zcorrected`` are in render.json and in its digest, so new or changed positions made init
  ask for ``--overwrite``. render.json still has the key ``zcorrected`` (now always false).
- ``_append`` (render.append) refused z-correction, from the config or from the volume's render.json::

      if old.get("zcorrected") or get(cfg, "zcorrect.enabled", False):
          log.error("render.append needs z-correction off: z-corrected planes interpolate between slices")
          return 1

- ``run`` needs no change: ``_Renderer.plane(sources, keep)`` renders any weighted plane, and ``keep`` (the
  next plane's sources) lets consecutive planes reuse the slice they share. Both are still in render.py.
- ``init`` now refuses a config with ``zcorrect.enabled: true``, and pipeline/preflight.py refuses it before
  anything is submitted: both checks go when the step comes back.
"""

import logging

import numpy as np
import pandas as pd

from pipeline.config import get, step_path

log = logging.getLogger(__name__)


def _positions(cfg, zs):
    """zcorrect positions (nm) of the selected z, or None when z-correction is off or not run."""
    if not get(cfg, "zcorrect.enabled", False):
        return None
    path = step_path(cfg, "zcorrect", "positions.csv")
    if not path.exists():
        log.warning("zcorrect.enabled but %s not found: rendering without z-correction", path)
        return None
    pos = pd.read_csv(path).set_index("z")["position_nm"].dropna()
    missing = [z for z in zs if z not in pos.index]
    if missing:
        raise ValueError(f"{path} has no position for {len(missing)} selected slices "
                         f"(e.g. z={missing[:3]}): re-run zcorrect")
    return pos.loc[zs].to_numpy(float)


def plane_sources(zs, f, positions=None, step_nm=None):
    """Source slices [(z, weight), ...] of every output plane.

    Without positions, plane k is the mean of the k-th group of f consecutive selected slices
    (neighbours by position in ``zs``, which may skip z). With positions (nm, one per z), planes
    are ``step_nm`` apart over the position range and interpolate linearly between the two
    slices that bracket them.
    """
    if positions is None:
        return [[(z, 1.0 / len(g)) for z in g] for g in (zs[i:i + f] for i in range(0, len(zs), f))]
    order = np.argsort(positions, kind="stable")
    p, zz = np.asarray(positions, float)[order], np.asarray(zs)[order]
    planes = []
    for k in range(int(np.floor((p[-1] - p[0]) / step_nm + 1e-6)) + 1):
        q = p[0] + k * step_nm
        i = int(np.searchsorted(p, q, side="right")) - 1   # p[i] <= q < p[i + 1]
        if i >= len(p) - 1:
            planes.append([(int(zz[-1]), 1.0)])
            continue
        t = (q - p[i]) / (p[i + 1] - p[i])
        planes.append([(int(z), float(w)) for z, w in ((zz[i], 1 - t), (zz[i + 1], t)) if w > 1e-6])
    return planes
