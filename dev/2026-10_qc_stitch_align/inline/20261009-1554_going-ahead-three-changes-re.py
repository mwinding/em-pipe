# Extracted from a session transcript (inline script, not saved as a file at the time).
# Time (UTC): 2026-10-09T15:54:10.098Z
# Ran: local
# Heredoc line: ${EM_PIPE_ENV}/bin/python - <<'EOF'
# Assistant note before it: Going ahead. Three changes for the re-render: - **Destreak:** on, at σ = 4, in the test config. - **Render memory:** raised from 32 GB to 64 GB, because destreak works on full-resolution tiles and four run at once. - **Clean re-render:** I'll clear the old render's plan file so it starts fresh, inst

import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
from pathlib import Path
p = Path(os.path.expandvars("configs/${DEV_PROJECT}_test_2day.yaml"))
s = p.read_text()
old_r = "render:\n  downsample: 4              # 32 nm voxels\n"
assert s.count(old_r) == 1
s = s.replace(old_r, """# Curtaining removal at the strength chosen from qc/destreak_compare_*.png (sigma 1/2/4/8 compared).
destreak:
  enabled: true
  sigma: 4.0

""" + old_r)
old_m = '  render: {array: 5, cpus: 8, mem: 32G, time: "04:00:00"}'
assert s.count(old_m) == 1
s = s.replace(old_m, '  render: {array: 5, cpus: 8, mem: 64G, time: "04:00:00"}')
p.write_text(s)
