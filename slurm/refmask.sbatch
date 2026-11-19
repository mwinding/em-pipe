#!/bin/bash
# refmask: reference tissue (neuropil) cells of every (z, tile) from the preview thumbnails (run: array over
# z-chunks), then merge -> output_dir/work/refmask/
# Usage: sbatch slurm/refmask.sbatch CONFIG run|merge [extra args]   (normally submitted by run_pipeline.sh)

#SBATCH --job-name=em-refmask
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

python -u -m pipeline.refmask --config "$CONFIG" "$@"
