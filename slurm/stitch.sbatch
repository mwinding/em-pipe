#!/bin/bash
# Stitch tiles within slices: 'run' (array over sampled slices), then 'merge' -> output_dir/stitch/
# Usage: sbatch slurm/stitch.sbatch CONFIG run|merge [extra args]   (normally submitted by run_pipeline.sh)

#SBATCH --job-name=em-stitch
#SBATCH --partition=ncpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mail-type=FAIL

set -eo pipefail
CONFIG=$1; shift
cd "${EM_PIPE_ROOT:-$SLURM_SUBMIT_DIR}"
source slurm/env.sh

python -u -m pipeline.stitch --config "$CONFIG" "$@"
