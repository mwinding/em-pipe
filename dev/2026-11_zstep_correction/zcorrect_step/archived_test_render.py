"""Tests of render's use of the zcorrect positions, taken out of tests/test_render.py when the step was removed
from the pipeline (26 Nov 2026). A record: pytest does not collect this file, and the tests cannot pass until
render reads positions.csv again (render_plane_plan.py, next to this file).

To use them again, put both tests back into tests/test_render.py: they use its helpers (``_setup``,
``_render``, ``_read``, ``_meta``, ``_aligned``, ``_corr``), its ``truth`` fixture and its constants ``SMALL``
and ``N``. ``_setup`` must then create ``work/zcorrect`` again (add "zcorrect" to the folders it makes).

``test_plane_sources_with_positions`` is the second half of ``test_plane_sources``; the first half (planes
without positions) is still in tests/test_render.py.
"""

import numpy as np
import pandas as pd
import pytest

from pipeline import render
from test_render import N, SMALL, _aligned, _corr, _meta, _read, _render, _setup   # tests/ on sys.path


def test_plane_sources_with_positions():
    planes = render.plane_sources([10, 11, 12], 1, np.array([0.0, 12.0, 16.0]), 8.0)
    assert planes[0] == [(10, 1.0)] and planes[2] == [(12, 1.0)]
    assert planes[1] == [(10, pytest.approx(1 / 3)), (11, pytest.approx(2 / 3))]
    # Positions out of z order: planes follow position, not z.
    assert render.plane_sources([10, 11, 12], 1, np.array([0.0, 16.0, 8.0]), 8.0) == [
        [(10, 1.0)], [(12, 1.0)], [(11, 1.0)]]


def test_zcorrect_plane_mapping(truth, tmp_path, monkeypatch):
    calls = []
    tile = render._Renderer._tile
    monkeypatch.setattr(render._Renderer, "_tile", lambda self, t: calls.append(t["z"]) or tile(self, t))
    zs = list(range(12))
    cfg, out = _setup(tmp_path, truth, zcorrect={"enabled": True}, selection={"z_end": zs[-1]})
    # Enabled but zcorrect not run yet: one plane per slice.
    assert render.main(["init", "--config", str(cfg)]) == 0
    assert len(_meta(out)["planes"]) == len(zs) and not _meta(out)["zcorrected"]
    rng = np.random.default_rng(4)
    pos = 8 * np.concatenate([[3.0], 3 + np.cumsum(rng.uniform(0.4, 1.7, N - 1))])
    pd.DataFrame({"z": range(N), "position_nm": pos}).to_csv(out / "work" / "zcorrect" / "positions.csv", index=False)
    assert render.main(["init", "--config", str(cfg)]) == 1      # inputs changed: needs --overwrite
    _render(cfg, "--overwrite")
    meta, vol = _meta(out), _read(out)
    p = pos[zs]
    assert meta["zcorrected"] and len(meta["planes"]) == int((p[-1] - p[0]) // 8) + 1 == len(vol)
    img, cov = _aligned(truth, zs)
    for k, src in enumerate(meta["planes"]):
        z, w = np.array([s[0] for s in src]), np.array([s[1] for s in src])
        assert w.sum() == pytest.approx(1) and np.dot(w, p[z]) == pytest.approx(p[0] + 8 * k)
        assert len(z) == 1 or z[1] == z[0] + 1   # bracketing slices are neighbours in position
        exp = 255 * np.tensordot(w, img[z], 1)
        ok = cov[z].all(axis=0)
        assert _corr(vol[k][ok], exp[ok]) > 0.99 and np.abs(vol[k][ok] - exp[ok]).mean() < 3
    # Consecutive planes share slices: each is rendered once per slab, not once per plane.
    slab = SMALL["slab"]
    per_slab = [{z for src in meta["planes"][k:k + slab] for z, _ in src} for k in range(0, len(vol), slab)]
    assert sorted(calls) == sorted(z for zz in per_slab for z in zz for _ in range(4))
