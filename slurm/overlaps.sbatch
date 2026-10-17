#!/bin/bash
# Overlaps: brightness relation of every overlapping tile pair, from the raw tiles (run: array over
# z-chunks), then merge -> output_dir/work/overlaps/relations.csv
# Usage: sbatch slurm/overlaps.sbatch CONFIG run|merge [extra args]   (normally submitted by run_pipeline.sh)
# A run task reads only the overlap strips (~0.4 s per left/right strip of a 13k px tile).

#SBATCH --job-name=em-overlaps
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

python -u -m pipeline.overlaps --config "$CONFIG" "$@"
