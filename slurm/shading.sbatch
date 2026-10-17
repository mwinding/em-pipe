#!/bin/bash
# Shading: flat-field per tile position and segment from the preview thumbnails -> output_dir/work/shading/
# Usage: sbatch slurm/shading.sbatch CONFIG [extra args]   (normally submitted by run_pipeline.sh)

#SBATCH --job-name=em-shading
#SBATCH --partition=ncpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mail-type=FAIL

set -eo pipefail
CONFIG=$1; shift
cd "${EM_PIPE_ROOT:-$SLURM_SUBMIT_DIR}"
source slurm/env.sh

python -u -m pipeline.shading --config "$CONFIG" "$@"
