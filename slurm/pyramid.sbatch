#!/bin/bash
# Pyramid: build one lower-resolution scale of the rendered volume, array over its shards
# Usage: sbatch --array=0-N slurm/pyramid.sbatch CONFIG run --scale S   (one job per scale, each after the previous)

#SBATCH --job-name=em-pyramid
#SBATCH --partition=ncpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mail-type=FAIL

set -eo pipefail
CONFIG=$1; shift
cd "${EM_PIPE_ROOT:-$SLURM_SUBMIT_DIR}"
source slurm/env.sh

python -u -m pipeline.pyramid --config "$CONFIG" "$@"
