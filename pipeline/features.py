"""SIFT feature matching shared by stitch and align.

detect → match (Lowe ratio test) → fit_model (RANSAC) is the same recipe Janelia's
render pipeline uses for FIB-SEM montage and slice-to-slice point matches.
"""

import cv2
import numpy as np

MODELS = ("translation", "rigid", "similarity", "affine")


def to_uint8(img, lo=None, hi=None, percentiles=(0.5, 99.5)):
    """Linear map to uint8; lo/hi default to robust percentiles of ``img``."""
    img = np.asarray(img, np.float32)
    if lo is None or hi is None:
        plo, phi = np.percentile(img, percentiles)
        lo = plo if lo is None else lo
        hi = phi if hi is None else hi
    scale = 255.0 / max(float(hi) - float(lo), 1e-6)
    return np.clip((img - lo) * scale, 0, 255).astype(np.uint8)


def downsample(img, factor):
    """Area-average downsample by an integer or float factor (factor 1 returns img)."""
    if factor == 1:
        return img
    h, w = img.shape
    size = (max(1, int(round(w / factor))), max(1, int(round(h / factor))))
    return cv2.resize(np.asarray(img, np.float32), size, interpolation=cv2.INTER_AREA)


def detect(img8, nfeatures=0, contrast_threshold=0.04, edge_threshold=10, sigma=1.6, mask=None):
    """SIFT keypoints of a uint8 image -> (xy float32 (N, 2), descriptors float32 (N, 128))."""
    sift = cv2.SIFT_create(nfeatures=nfeatures, contrastThreshold=contrast_threshold,
                           edgeThreshold=edge_threshold, sigma=sigma)
    kps, desc = sift.detectAndCompute(img8, mask)
    if desc is None or not kps:
        return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32)
    return np.array([k.pt for k in kps], np.float32), desc.astype(np.float32)


def match(desc_a, desc_b, ratio=0.8):
    """Indices (ia, ib) of matches passing Lowe's ratio test."""
    if len(desc_a) < 2 or len(desc_b) < 2:
        return np.zeros(0, int), np.zeros(0, int)
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(desc_a, desc_b, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < ratio * p[1].distance]
    return np.array([m.queryIdx for m in good], int), np.array([m.trainIdx for m in good], int)


def _estimate(model, pa, pb):
    """Least-squares model mapping pa -> pb (needs at least _MIN_POINTS[model] points)."""
    if model == "translation":
        t = (pb - pa).mean(axis=0)
        return np.array([[1.0, 0.0, t[0]], [0.0, 1.0, t[1]]])
    if model in ("rigid", "similarity"):
        # Umeyama: rotation (+ isotropic scale for similarity) about the centroids.
        ca, cb = pa.mean(axis=0), pb.mean(axis=0)
        qa, qb = pa - ca, pb - cb
        U, S, Vt = np.linalg.svd(qb.T @ qa)
        D = np.diag([1.0, np.sign(np.linalg.det(U @ Vt)) or 1.0])
        R = U @ D @ Vt
        s = 1.0
        if model == "similarity":
            var = (qa ** 2).sum()
            s = float((S * np.diag(D)).sum() / var) if var > 0 else 1.0
        return np.hstack([s * R, (cb - s * R @ ca)[:, None]])
    if model == "affine":
        X = np.hstack([pa, np.ones((len(pa), 1))])
        sol, *_ = np.linalg.lstsq(X, pb, rcond=None)
        return sol.T
    raise ValueError(f"unknown model {model!r}; expected one of {MODELS}")


_MIN_POINTS = {"translation": 1, "rigid": 2, "similarity": 2, "affine": 3}


def fit_model(pa, pb, model="translation", threshold=3.0, iterations=1000, min_inliers=10, seed=0):
    """RANSAC fit of ``model`` mapping pa -> pb (both (N, 2) in pixels).

    Returns (A 2×3, inlier bool mask), or (None, mask) if fewer than ``min_inliers`` agree.
    The returned model is a least-squares refit on the final inliers.
    """
    if model not in _MIN_POINTS:
        raise ValueError(f"unknown model {model!r}; expected one of {MODELS}")
    pa = np.asarray(pa, np.float64).reshape(-1, 2)
    pb = np.asarray(pb, np.float64).reshape(-1, 2)
    n, k = len(pa), _MIN_POINTS[model]
    if n < max(k, min_inliers):
        return None, np.zeros(n, bool)
    rng = np.random.default_rng(seed)
    best = np.zeros(n, bool)
    for _ in range(iterations):
        idx = rng.choice(n, size=k, replace=False)
        with np.errstate(all="ignore"):
            A = _estimate(model, pa[idx], pb[idx])
        if not np.all(np.isfinite(A)):
            continue
        inl = residuals(A, pa, pb) < threshold
        if inl.sum() > best.sum():
            best = inl
            if best.sum() > 0.95 * n:
                break
    if best.sum() < min_inliers:
        return None, best
    # Refit on inliers, then re-select inliers under the refit model once.
    A = _estimate(model, pa[best], pb[best])
    inl = residuals(A, pa, pb) < threshold
    if inl.sum() >= best.sum():
        best = inl
        A = _estimate(model, pa[best], pb[best])
    return A, best


def residuals(A, pa, pb):
    """Per-point distance between A(pa) and pb."""
    return np.linalg.norm(np.asarray(pa) @ A[:, :2].T + A[:, 2] - np.asarray(pb), axis=1)
