# Extracted from a session transcript (inline script, not saved as a file at the time).
# Time (UTC): 2026-10-18T21:32:03.138Z
# Ran: local
# Heredoc line: (whole command)
# Assistant note before it: Yes, for setting the black and white points it is. You've put your finger on the cause. **Why it's like that:** the preview step reads every raw tile once and keeps 1/16-scale versions for statistics, thumbnails and QC. The window percentiles were taken from those because it was cheap and the window

python3 -c "
from PIL import Image
im = Image.open('qc/35h_shading/clipping/neuropil_0_z4000.png')
print(im.size, im.mode)
# panels: 1024 wide + 16 gap; 40 px label on top
x0, y0, s = 560, 400, 256
crops = [im.crop((p*1040 + x0, 40 + y0, p*1040 + x0 + s, 40 + y0 + s)).resize((s*3, s*3), Image.NEAREST) for p in range(4)]
out = Image.new('L', (4*s*3 + 3*12, s*3), 255)
for i, c in enumerate(crops): out.paste(c, (i*(s*3+12), 0))
out.save('${DEV_WORK}/z4000_zoom.png')
import numpy as np
a = np.array(im)
for p in range(4):
    b = a[40+y0:40+y0+s, p*1040+x0:p*1040+x0+s]
    print(p, np.percentile(b, [1,5,25,50,75,95,99]).round())
"
