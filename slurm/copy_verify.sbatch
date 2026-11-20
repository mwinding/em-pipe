#!/bin/bash
# One step of tools/copy_verify.py (copy, md5 or verify) as a Slurm job; array jobs take file K = task id.
# Normally submitted as a chain by:  python tools/copy_verify.py submit SRC DST [--pattern REGEX] [--group-rw]
# Usage: sbatch [--array=0-(N-1)] slurm/copy_verify.sbatch STEP SRC DST [options]

#SBATCH --job-name=copy-verify
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --mail-type=FAIL

set -eo pipefail
cd "${EM_PIPE_ROOT:-$SLURM_SUBMIT_DIR}"
source slurm/env.sh

python -u tools/copy_verify.py "$@"
