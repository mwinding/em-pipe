# Extracted from a session transcript (inline script, not saved as a file at the time).
# Time (UTC): 2026-10-15T20:03:57.101Z
# Ran: remote, via ssh
# Heredoc line: cd ${DEV_WORK} && /usr/bin/python3 - <<'EOF'
# Assistant note before it: I'll switch to one ImageJ TIFF stack per spot, which suits Fiji better: - **Layout:** the same 500 × 500 px centre crops of every tile, laid out like the tiles. - **Slices:** one per raw slice (no back-and-forth), with full 256 grey levels. - **Labels:** each slice's label (shown in Fiji) gives the 

from pathlib import Path
p = Path("spot_gifs.py"); s = p.read_text()
def rep(old, new):
    global s
    assert s.count(old) == 1, old[:60]
    s = s.replace(old, new)
rep('''"""Spot-check GIFs of a run's flagged stretches: a CROP x CROP px crop at full resolution from the
centre of every tile (laid out like the tiles), stepping through the slices and back.''',
    '''"""Spot-check stacks of a run's flagged stretches: a CROP x CROP px crop at full resolution from the
centre of every tile (laid out like the tiles), one ImageJ TIFF per spot with a slice per z.''')
rep('''blend) but without CLAHE, so contrast doesn't change from crop to crop; 64 grey levels.''',
    '''blend) but without CLAHE, so contrast doesn't change from crop to crop. Slice labels give z, time
and "<" on flagged slices; calibrated in microns.''')
rep('''import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont''', '''import numpy as np
import pandas as pd
import tifffile''')
rep('''try:
    font = ImageFont.load_default(size=15)
except TypeError:
    font = ImageFont.load_default()
''', '')
rep('''        frames = []
        for z in span:
            grid = Image.new("L", (nc * CROP + (nc - 1) * 4, 26 + nr * CROP + (nr - 1) * 4), 0)
            for tile, rr in renderers.items():
                a, b = map(int, tile.split("-"))
                grid.paste(Image.fromarray(rr.plane([(int(z), 1.0)])), (b * (CROP + 4), 26 + a * (CROP + 4)))
            draw = ImageDraw.Draw(grid)
            flag = " <" if s["z0"] <= z <= s["z1"] else ""
            draw.text((6, 4), f"spot {k}  z {int(z)}  {stamp.loc[int(z)]:%d %b %H:%M:%S}{flag}   {why}", fill=255, font=font)
            frames.append(grid.point(lambda v: v & 0xFC))   # 64 grey levels: ~half the GIF size
        seq = frames + frames[-2:0:-1]
        name = f"spot{k:02d}_z{int(span[0])}-{int(span[-1])}.gif"
        seq[0].save(out / name, save_all=True, append_images=seq[1:], duration=FRAME_MS, loop=0)''',
    '''        stack = np.zeros((len(span), nr * CROP + (nr - 1) * 4, nc * CROP + (nc - 1) * 4), np.uint8)
        labels = []
        for i, z in enumerate(span):
            for tile, rr in renderers.items():
                a, b = map(int, tile.split("-"))
                stack[i, a * (CROP + 4):a * (CROP + 4) + CROP, b * (CROP + 4):b * (CROP + 4) + CROP] = rr.plane([(int(z), 1.0)])
            flag = " <" if s["z0"] <= z <= s["z1"] else ""
            labels.append(f"z {int(z)} {stamp.loc[int(z)]:%d %b %H:%M:%S}{flag} | spot {k}: {why}")
        name = f"spot{k:02d}_z{int(span[0])}-{int(span[-1])}.tif"
        um = 0.001 * float(cfg["check"]["voxel_size_nm"][0]) if cfg["check"].get("voxel_size_nm") else 0.008
        tifffile.imwrite(out / name, stack, imagej=True, resolution=(1 / um, 1 / um),
                         metadata={"spacing": um, "unit": "micron", "axes": "ZYX", "Labels": labels})''')
rep('''CROP, PAD, MERGE, JUMP, FRAME_MS = 500, 6, 15, 15.0, 150''', '''CROP, PAD, MERGE, JUMP = 500, 6, 15, 15.0''')
p.write_text(s)
Path("spot_tiffs.py").write_text(s)
