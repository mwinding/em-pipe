#!/bin/bash
# refstats: reference tissue median and lc32 on the common scale at sampled slices (run: array over chunks of
# sampled slices; reads whole tile slices at full resolution), then merge -> output_dir/work/refstats/
# Usage: sbatch slurm/refstats.sbatch CONFIG run|merge [extra args]   (normally submitted by run_pipeline.sh)

#SBATCH --job-name=em-refstats
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mail-type=FAIL

set -eo pipefail
CONFIG=$1; shift
cd "${EM_PIPE_ROOT:-$SLURM_SUBMIT_DIR}"
source slurm/env.sh

python -u -m pipeline.refstats --config "$CONFIG" "$@"
