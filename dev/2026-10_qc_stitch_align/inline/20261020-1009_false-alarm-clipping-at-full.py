# Extracted from a session transcript (inline script, not saved as a file at the time).
# Time (UTC): 2026-10-20T10:09:10.682Z
# Ran: remote, via ssh
# Heredoc line: cd ${DEV_WORK}/cluster && python3 - <<'EOF'
# Assistant note before it: A false alarm on clipping. At full resolution the 37i raw data isn't clipped. The brightest pixels are isolated single values around 65,200, with nothing piling up at a ceiling. The apparent cut-off near 64,400 came from the preview statistics, which are measured on 1/16 thumbnails, where averaging 

from pathlib import Path
p = Path("junction.py")
s = p.read_text()
old = '''s0, s3 = omezarr.open_scale(root, 0), omezarr.open_scale(root, 3)
J, W = 1913, 1024
m = (s3[J // 8].read().result() > 0) & (s3[(J + 1) // 8].read().result() > 0)
m = cv2.erode(m.astype(np.uint8), np.ones((40, 40), np.uint8)) > 0   # >= 2.5k px inside both
rr, cc = np.nonzero(m)
r0, r1, c0, c1 = rr.min(), rr.max(), cc.min(), cc.max()
print(f"common area at s3: rows {r0}-{r1}, cols {c0}-{c1} ({m.sum()} px)")
cent = [(int(8 * (r0 + (r1 - r0) * fr)), int(8 * (c0 + (c1 - c0) * fc))) for fr in (0.2, 0.5, 0.8) for fc in (0.2, 0.5, 0.8)]
cent = [(r, c) for r, c in cent if m[r // 8, c // 8]]'''
new = '''s0 = omezarr.open_scale(root, 0)
J, W, S = 1913, 1024, 16
# data masks of the two 8 nm planes, sampled every S px (s1-s6 also average over z)
m = (s0[J, ::S, ::S].read().result() > 0) & (s0[J + 1, ::S, ::S].read().result() > 0)
m = cv2.erode(m.astype(np.uint8), np.ones((2 * W // S, 2 * W // S), np.uint8)) > 0
rr, cc = np.nonzero(m)
r0, r1, c0, c1 = rr.min(), rr.max(), cc.min(), cc.max()
print(f"area with data in both planes (8 nm px): rows {S * r0}-{S * r1}, cols {S * c0}-{S * c1}")
cent = [(int(S * (r0 + (r1 - r0) * fr)), int(S * (c0 + (c1 - c0) * fc))) for fr in (0.2, 0.5, 0.8) for fc in (0.2, 0.5, 0.8)]
cent = [(r, c) for r, c in cent if m[r // S, c // S]]
print(f"{len(cent)} windows of {W} px")'''
assert s.count(old) == 1
s = s.replace(old, new)
old2 = '''    for a, b in pairs:
        (dx, dy), resp = cv2.phaseCorrelate(planes[a], planes[b], win)'''
new2 = '''    if min((pl > 0).mean() for pl in planes.values()) < 0.99:
        print(f"window at {r},{c} skipped: not all data in every plane")
        continue
    for a, b in pairs:
        (dx, dy), resp = cv2.phaseCorrelate(planes[a], planes[b], win)'''
assert s.count(old2) == 1
p.write_text(s.replace(old2, new2))
