#!/bin/bash
# Render the corrected volume (scale 0) -> output_dir/render/<render.name>
# Usage: sbatch slurm/render.sbatch CONFIG init        (single job: plan + empty volume)
#        sbatch --array=0-N slurm/render.sbatch CONFIG run   (array over z-slabs; normally via run_pipeline.sh)
#
# Resources are sized for `run` on the real data at render.downsample 1. Measured peak memory per
# plane pixel (canvas = union of all slices, ~700 Mpx or more with drift): slab buffer 64 B (render.slab
# planes of uint8) + rendering 31 B (translation only), 47 B (rotated: warp path), 54 B (zcorrect) or
# 62 B (both) + ~13 B with destreak + ~4 B for writing shards one row at a time. At 700 Mpx that is
# 70-100 GB; 160G covers canvases up to ~1.1 Gpx in the heaviest mode. Memory scales with
# render.slab / render.downsample^2. A slab takes roughly 5-10 min, so time depends on slabs per
# task (= ceil(planes / slab) / array size). init needs only a few GB (e.g. --mem=16G).

#SBATCH --job-name=em-render
#SBATCH --partition=ncpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mail-type=FAIL

set -eo pipefail
CONFIG=$1; shift
cd "${EM_PIPE_ROOT:-$SLURM_SUBMIT_DIR}"
source slurm/env.sh

python -u -m pipeline.render --config "$CONFIG" "$@"
