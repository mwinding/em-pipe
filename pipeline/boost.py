"""Small gradient-boosted decision trees for binary classification (logistic loss), numpy only.

Features are binned into at most ``bins`` quantile bins (edges kept in the model); each tree has a
fixed depth and is grown on histograms of gradients and hessians, as in LightGBM/XGBoost "hist".
Deterministic (no subsampling), so a model rebuilt from the same data is the same. A model is a
plain dict (JSON-serialisable): {"names", "edges", "trees", "base", "rate"}; a tree is a list of
levels, each a list of (feature, bin) splits, and its leaf values.
"""

import numpy as np


def _bin(X, edges):
    return np.stack([np.searchsorted(e, X[:, j], side="right") for j, e in enumerate(edges)], axis=1).astype(np.int16)


def fit(X, y, w=None, names=None, trees=150, depth=3, rate=0.1, bins=32, l2=1.0, min_hess=5.0):
    """Fit on X (n, f), y in {0, 1}, sample weights w. Returns the model dict."""
    X, y = np.asarray(X, np.float64), np.asarray(y, np.float64)
    w = np.ones(len(y)) if w is None else np.asarray(w, np.float64)
    edges = []
    for j in range(X.shape[1]):
        q = np.unique(np.quantile(X[:, j], np.linspace(0, 1, bins + 1)[1:-1]))
        edges.append(q)
    B = _bin(X, edges)
    nb = max(len(e) for e in edges) + 1
    p0 = np.clip(np.average(y, weights=w), 1e-3, 1 - 1e-3)
    base = float(np.log(p0 / (1 - p0)))
    F = np.full(len(y), base)
    out = []
    for _ in range(trees):
        p = 1 / (1 + np.exp(-F))
        g, h = w * (p - y), w * np.maximum(p * (1 - p), 1e-6)
        node = np.zeros(len(y), np.int64)
        levels = []
        for d in range(depth):
            splits = []
            new = np.zeros_like(node)
            for k in range(2 ** d):
                sel = node == k
                best = (0.0, 0, nb)   # gain, feature, bin (bin nb: no split, all go left)
                if h[sel].sum() >= 2 * min_hess:
                    G, H = g[sel].sum(), h[sel].sum()
                    for j in range(B.shape[1]):
                        gs = np.bincount(B[sel, j], g[sel], nb)
                        hs = np.bincount(B[sel, j], h[sel], nb)
                        gl, hl = np.cumsum(gs)[:-1], np.cumsum(hs)[:-1]
                        ok = (hl >= min_hess) & (H - hl >= min_hess)
                        if not ok.any():
                            continue
                        gain = gl ** 2 / (hl + l2) + (G - gl) ** 2 / (H - hl + l2) - G ** 2 / (H + l2)
                        gain[~ok] = -np.inf
                        i = int(np.argmax(gain))
                        if gain[i] > best[0]:
                            best = (float(gain[i]), j, i)
                _, j, b = best
                splits.append((int(j), int(b)))
                right = sel & (B[:, j] > b)
                new[sel] = 2 * k + right[sel]
            node = new
            levels.append(splits)
        G = np.bincount(node, g, 2 ** depth)
        H = np.bincount(node, h, 2 ** depth)
        leaf = -G / (H + l2)
        F += rate * leaf[node]
        out.append({"levels": levels, "leaf": leaf.tolist()})
    return {"names": list(names) if names is not None else None, "edges": [e.tolist() for e in edges],
            "trees": out, "base": base, "rate": rate}


def predict(model, X):
    """Probability of class 1 for X (n, f)."""
    X = np.asarray(X, np.float64)
    B = _bin(X, [np.asarray(e) for e in model["edges"]])
    F = np.full(len(X), model["base"])
    idx = np.arange(len(X))
    for t in model["trees"]:
        node = np.zeros(len(X), np.int64)
        for splits in t["levels"]:
            s = np.asarray(splits)
            j, b = s[node, 0], s[node, 1]
            node = 2 * node + (B[idx, j] > b)
        F += model["rate"] * np.asarray(t["leaf"])[node]
    return 1 / (1 + np.exp(-F))
