#!/bin/bash
# Upload the precomputed export (or upload.source) to a Google Cloud Storage bucket: run (array) or finish
# Usage: sbatch [--array=0-N] slurm/upload.sbatch CONFIG run|finish [--dry-run]   (normally run_pipeline.sh)
# Needs the service-account key at upload.credentials (default ~/.cloudvolume/secrets/google-secret.json).

#SBATCH --job-name=em-upload
#SBATCH --partition=ncpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mail-type=FAIL

set -eo pipefail
CONFIG=$1; shift
cd "${EM_PIPE_ROOT:-$SLURM_SUBMIT_DIR}"
source slurm/env.sh

python -u -m pipeline.upload --config "$CONFIG" "$@"
