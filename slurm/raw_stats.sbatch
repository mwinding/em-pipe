#!/bin/bash
# Raw detector statistics per tile by class (tools/raw_stats.py): reads the raw stacks at sampled slices.
# Usage:
#   one job:   sbatch slurm/raw_stats.sbatch CONFIG [--slices N | --z Z,...] [--out DIR]
#   in parallel (task k of n takes every n-th sampled slice), then merge:
#     A=$(sbatch --parsable --array=0-7 slurm/raw_stats.sbatch CONFIG --num-tasks 8 [--slices N])
#     sbatch --dependency=afterok:$A slurm/raw_stats.sbatch CONFIG merge [--out DIR]
# Outputs: <output_dir>/qc/raw_stats/ (or --out). Read-only on the data.

#SBATCH --job-name=em-raw-stats
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mail-type=FAIL

set -eo pipefail
CONFIG=$1; shift
cd "${EM_PIPE_ROOT:-$SLURM_SUBMIT_DIR}"
source slurm/env.sh

if [ "${1:-}" = merge ]; then
    shift
    python -u tools/raw_stats.py merge --config "$CONFIG" "$@"
else
    python -u tools/raw_stats.py run --config "$CONFIG" "$@"
fi
