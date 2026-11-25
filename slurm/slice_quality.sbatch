#!/bin/bash
# Per-slice quality flags (tools/slice_quality.py): reads three bands of every raw tile slice.
# Usage (task k of n takes every n-th chunk of 50 slices), then merge:
#   A=$(sbatch --parsable --array=0-19 slurm/slice_quality.sbatch CONFIG)
#   sbatch --dependency=afterok:$A slurm/slice_quality.sbatch CONFIG merge
# Outputs: <output_dir>/qc/slice_quality.csv, slice_flags.csv, slice_quality.png, slice_quality/ (or --out DIR).
# Read-only on the data; flags only, nothing is excluded.

#SBATCH --job-name=em-slice-quality
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --mail-type=FAIL

set -eo pipefail
CONFIG=$1; shift
cd "${EM_PIPE_ROOT:-$SLURM_SUBMIT_DIR}"
source slurm/env.sh

if [ "${1:-}" = merge ]; then
    shift
    python -u tools/slice_quality.py merge --config "$CONFIG" "$@"
else
    python -u tools/slice_quality.py run --config "$CONFIG" "$@"
fi
