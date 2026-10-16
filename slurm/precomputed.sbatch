#!/bin/bash
# Neuroglancer precomputed export of the rendered volume: init, run (array over shards) or verify
# Usage: sbatch [--array=0-N] slurm/precomputed.sbatch CONFIG init|run|verify   (normally run_pipeline.sh)

#SBATCH --job-name=em-precomputed
#SBATCH --partition=ncpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mail-type=FAIL

set -eo pipefail
CONFIG=$1; shift
cd "${EM_PIPE_ROOT:-$SLURM_SUBMIT_DIR}"
source slurm/env.sh

python -u -m pipeline.precomputed --config "$CONFIG" "$@"
