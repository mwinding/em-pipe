#!/bin/bash
# Check the raw data: inventory, faults, global slice list -> output_dir/check/
# Usage: sbatch slurm/check.sbatch CONFIG [extra args]   (normally submitted by run_pipeline.sh)

#SBATCH --job-name=em-check
#SBATCH --partition=ncpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mail-type=FAIL

set -eo pipefail
CONFIG=$1; shift
cd "${EM_PIPE_ROOT:-$SLURM_SUBMIT_DIR}"
source slurm/env.sh

python -u -m pipeline.check --config "$CONFIG" "$@"
