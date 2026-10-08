#!/bin/bash
# Slice-to-slice alignment -> output_dir/align/
# Usage: sbatch slurm/align.sbatch CONFIG run|solve [extra args]   (normally submitted by run_pipeline.sh)
#   run:   array job; SIFT matches per chunk of align.chunk_slices z (~20-30 min per 100 slices, ~2 GB)
#   solve: single job; global least-squares solve over all chunks (~11,500 slices: <1 min, ~5 GB)

#SBATCH --job-name=em-align
#SBATCH --partition=ncpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mail-type=FAIL

set -eo pipefail
CONFIG=$1; shift
cd "${EM_PIPE_ROOT:-$SLURM_SUBMIT_DIR}"
source slurm/env.sh

python -u -m pipeline.align --config "$CONFIG" "$@"
