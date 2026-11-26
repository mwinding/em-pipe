#!/bin/bash
# Axial (slice thickness) correction -> output_dir/zcorrect/
# Usage: sbatch [--array=0-N] S/zcorrect.sbatch CONFIG run   (array: NCC per z-chunk)
#        sbatch S/zcorrect.sbatch CONFIG solve               (single: positions.csv)
# (S = dev/2026-11_zstep_correction/zcorrect_step, from the repository root. Was slurm/zcorrect.sbatch,
# submitted by run_pipeline.sh, until the step was removed from the pipeline on 26 Nov 2026)
# A run task reads one crop_px square per tile per slice (2 MB at 1024 px, spread over a ~28 MB span
# of a 2x2-grid file) and keeps max_distance slices of crops in memory; solve takes minutes and < 1 GB.

#SBATCH --job-name=em-zcorrect
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

python -u dev/2026-11_zstep_correction/zcorrect_step/zcorrect.py --config "$CONFIG" "$@"
