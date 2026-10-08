"""2D affine transforms as 2×3 arrays ``[[a, b, tx], [c, d, ty]]``: dst = A @ [x, y, 1].

Conventions (see docs/design.md): points are (x, y); stitch maps tile → montage,
align maps montage → aligned; render applies ``compose(align, stitch)``.
"""

import numpy as np
import pandas as pd

COLUMNS = ["a", "b", "tx", "c", "d", "ty"]


def identity():
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def translation(tx, ty):
    return np.array([[1.0, 0.0, tx], [0.0, 1.0, ty]])


def _h(A):
    return np.vstack([np.asarray(A, float), [0.0, 0.0, 1.0]])


def compose(outer, inner):
    """Transform that applies ``inner`` first, then ``outer``."""
    return (_h(outer) @ _h(inner))[:2]


def invert(A):
    return np.linalg.inv(_h(A))[:2]


def apply(A, pts):
    """Apply to an (N, 2) array of (x, y) points."""
    pts = np.asarray(pts, float).reshape(-1, 2)
    A = np.asarray(A, float)
    return pts @ A[:, :2].T + A[:, 2]


def is_translation(A, tol=1e-9):
    A = np.asarray(A, float)
    return np.allclose(A[:, :2], np.eye(2), atol=tol)


def corners(width, height):
    """Corner points of a width×height image, in pixel-edge coordinates."""
    return np.array([[0, 0], [width, 0], [0, height], [width, height]], float)


def bbox(A, width, height):
    """(xmin, ymin, xmax, ymax) of a transformed width×height image."""
    p = apply(A, corners(width, height))
    return p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()


def to_row(A):
    return dict(zip(COLUMNS, np.asarray(A, float).ravel()))


def from_row(row):
    return np.array([[row["a"], row["b"], row["tx"]], [row["c"], row["d"], row["ty"]]], float)


def read_csv(path, key_cols):
    """Read a transforms CSV into {key: A}; key is a tuple when several key columns are given."""
    df = pd.read_csv(path, dtype={"tile": str})
    keys = df[key_cols[0]] if len(key_cols) == 1 else list(df[key_cols].itertuples(index=False, name=None))
    mats = df[COLUMNS].to_numpy(float).reshape(-1, 2, 3)
    return dict(zip(keys if len(key_cols) > 1 else keys.tolist(), mats))


def to_frame(records):
    """records: iterable of (dict_of_key_columns, A) -> DataFrame with key columns + COLUMNS."""
    rows = [{**keys, **to_row(A)} for keys, A in records]
    return pd.DataFrame(rows)
