#!/bin/bash
# Quick-look TIFF of the rendered volume for Fiji -> output_dir/<volume stem>_<voxel>nm.tif
# Usage: sbatch slurm/export.sbatch CONFIG [extra args]   (normally submitted by run_pipeline.sh)

#SBATCH --job-name=em-export
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

python -u -m pipeline.export --config "$CONFIG" "$@"
