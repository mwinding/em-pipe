#!/bin/bash
# Intensity: smoothed, tile-balanced display levels per (z, tile) -> output_dir/intensity/
# Usage: sbatch slurm/intensity.sbatch CONFIG [extra args]   (normally submitted by run_pipeline.sh)

#SBATCH --job-name=em-intensity
#SBATCH --partition=ncpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mail-type=FAIL

set -eo pipefail
CONFIG=$1; shift
cd "${EM_PIPE_ROOT:-$SLURM_SUBMIT_DIR}"
source slurm/env.sh

python -u -m pipeline.intensity --config "$CONFIG" "$@"
