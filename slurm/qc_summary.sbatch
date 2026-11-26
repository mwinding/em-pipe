#!/bin/bash
# The last job of a chain: the QC page of the volume (tools/qc_summary.py), after the flicker the levels
# predict (tools/zflicker.py --config; no volume is read).
# Usage: sbatch slurm/qc_summary.sbatch CONFIG [qc_summary.py options]   (normally submitted by run_pipeline.sh,
# which makes it wait for every job of the chain, whether they succeeded or not, and passes --chain FILE: the
# page then says where a chain stopped and why).

#SBATCH --job-name=em-qc-summary
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mail-type=FAIL

set -eo pipefail
CONFIG=$1; shift
cd "${EM_PIPE_ROOT:-$SLURM_SUBMIT_DIR}"
source slurm/env.sh

python -u tools/qc_summary.py "$CONFIG" --predict-flicker "$@"
