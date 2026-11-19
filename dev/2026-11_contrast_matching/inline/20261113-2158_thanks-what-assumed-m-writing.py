# Extracted from a session transcript (inline script, not saved as a file at the time).
# Time (UTC): 2026-11-13T21:58:08.564Z
# Ran: local
# Heredoc line: cd ${DEV_WORK} && ${EM_PIPE_ENV}/bin/python - <<'EOF'
# Assistant note before it: Thanks, that's what I assumed. I'm writing the measurement job now.

import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import re
s = open("clahe_match.sbatch").read()
start = s.index("def crops_for(")
end = s.index('json.dump(rows')
new = os.path.expandvars('''def crops_for(p, P, z, n):
    """Up to n tile-aligned K x K crops (unrotated tile indices) whose inner part is >= 97 % tissue."""
    H, W = P.shape[1:]
    (gx, tx), (gy, ty) = grid(W), grid(H)
    lo = np.asarray(opn(p, 4)[z // 16].read().result())[::-1, ::-1]   # unrotated, 128 nm
    bg = background.mask(lo) | ndimage.binary_dilation(lo == 0, iterations=4)
    f = W / lo.shape[1]
    ok = []
    for iy in range(0, gy - K):
        for ix in range(0, gx - K):
            y0, y1 = int((iy + 1) * ty / f), int((iy + K - 1) * ty / f)
            x0, x1 = int((ix + 1) * tx / f), int((ix + K - 1) * tx / f)
            if 1 - bg[y0:y1, x0:x1].mean() >= 0.97:
                ok.append((iy, ix))
    picked = []
    for i in rng.permutation(len(ok)):
        iy, ix = ok[i]
        if all(abs(iy - a) >= K or abs(ix - b) >= K for a, b in picked):
            picked.append((iy, ix))
        if len(picked) == n:
            break
    return picked, (tx, ty), (H, W)


def read_crops(arr, za, zb, iy, ix, tx, ty, H, W):
    """Unrotated crops of K x K CLAHE tiles at tile (iy, ix), planes za and zb (one read: same z chunk),
    from a volume stored rotated 180 degrees."""
    ys, xs = slice(H - (iy + K) * ty, H - iy * ty), slice(W - (ix + K) * tx, W - ix * tx)
    b = np.asarray(arr[za:zb + 1:zb - za, ys, xs].read().result())
    return [b[i, ::-1, ::-1] for i in range(2)]


sets = {"35h": [], "37i": []}
v35p, v35c = f"{S35}/${DEV_PREFIX}35h_8nm.ome.zarr", f"{S35}/${DEV_PREFIX}35h_8nm_clahe.ome.zarr"
n35 = opn(v35p, 0).shape[0]
for frac in (0.2, 0.5, 0.8):
    z0 = 64 * round(n35 * frac / 64)
    sets["35h"].append((v35p, v35c, z0 + 16, z0 + 48))
for d in (1000, 9600, 16500):
    sets["37i"].append((f"{S37}/${DEV_PREFIX}37i_test_z{d}_8nm.ome.zarr",
                        f"{S37}/${DEV_PREFIX}37i_test_z{d}_8nm_clahe.ome.zarr", 16, 48))

rows, panels = [], {}
for name, items in sets.items():
    for p, c, za, zb in items:
        P, C = opn(p, 0), opn(c, 0)
        picked, (tx, ty), (H, W) = crops_for(p, P, za, NCROP)
        print(f"{name} {os.path.basename(p)} z {za},{zb}: {len(picked)} crops, CLAHE tile {tx}x{ty}", flush=True)
        inner = (slice(ty, -ty), slice(tx, -tx))
        for iy, ix in picked:
            for z, plain, vol in zip((za, zb), read_crops(P, za, zb, iy, ix, tx, ty, H, W),
                                     read_crops(C, za, zb, iy, ix, tx, ty, H, W)):
                rec = {"set": name, "vol": os.path.basename(p), "z": z, "iy": iy, "ix": ix,
                       "plain": metrics(plain[inner]), "volume_clahe": metrics(vol[inner])}
                outs = {cl: cv2.createCLAHE(clipLimit=cl, tileGridSize=(K, K)).apply(plain) for cl in CLIPS}
                diff = np.abs(outs[1.5][inner].astype(int) - vol[inner])
                rec["repro_exact_frac"], rec["repro_max_diff"] = float((diff == 0).mean()), int(diff.max())
                for cl, out in outs.items():
                    rec[f"clip{cl}"] = metrics(out[inner])
                rows.append(rec)
                k = (name, sum(r["set"] == name for r in rows))
                if z == za and len([q for q in panels if q[0] == name]) < 3:   # 1024 px from the middle
                    c0y, c0x = (plain.shape[0] - 1024) // 2, (plain.shape[1] - 1024) // 2
                    sl = (slice(c0y, c0y + 1024), slice(c0x, c0x + 1024))
                    panels[k] = {"plain": plain[sl], **{f"clip{cl}": o[sl] for cl, o in outs.items()},
                                 "label": f"{rec['vol'].replace('.ome.zarr', '')} z{z}"}

''')
s = s[:start] + new + s[end:]
s = s.replace("plain_root = {}\n", "")
open("clahe_match.sbatch", "w").write(s)
