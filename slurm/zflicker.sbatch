#!/bin/bash
# tools/zflicker.py as a Slurm job: slice-to-slice brightness flicker of rendered volumes (and the raw tiles).
# Usage: sbatch slurm/zflicker.sbatch --out DIR [--config C] NAME=VOLUME [NAME=VOLUME ...]

#SBATCH --job-name=zflicker
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mail-type=FAIL

set -eo pipefail
cd "${EM_PIPE_ROOT:-$SLURM_SUBMIT_DIR}"
source slurm/env.sh

python -u tools/zflicker.py "$@"
