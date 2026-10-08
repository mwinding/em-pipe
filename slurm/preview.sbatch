#!/bin/bash
# Preview: per-slice stats + thumbnails (run: array over z-chunks), then merge -> output_dir/preview/
# Usage: sbatch slurm/preview.sbatch CONFIG run|merge [extra args]   (normally submitted by run_pipeline.sh)
# A run task reads one ~354 MB tile slice at a time; ~70 GB of raw reads per 50-slice chunk of a 2x2 grid.

#SBATCH --job-name=em-preview
#SBATCH --partition=ncpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mail-type=FAIL

set -eo pipefail
CONFIG=$1; shift
cd "${EM_PIPE_ROOT:-$SLURM_SUBMIT_DIR}"
source slurm/env.sh

python -u -m pipeline.preview --config "$CONFIG" "$@"
