#!/bin/bash
# One step of tools/split_stack.py (split, check or verify) as a Slurm job; array jobs take split K = task id.
# Normally submitted as a chain by:  python tools/split_stack.py submit JOINED_DIR REF_DIR [options]
# Usage: sbatch [--array=0-(N-1)] slurm/split_stack.sbatch STEP JOINED_DIR REF_DIR [options]

#SBATCH --job-name=split-stack
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --mail-type=FAIL

set -eo pipefail
cd "${EM_PIPE_ROOT:-$SLURM_SUBMIT_DIR}"
source slurm/env.sh

python -u tools/split_stack.py "$@"
