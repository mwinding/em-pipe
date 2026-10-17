#!/bin/bash
# Acquisition movie (the stack built up slice by slice, then the fly-through) -> output_dir/<movie name>.mp4
# (settings: the config's movie: and build:)
# Usage: sbatch --output=LOG slurm/movie_build.sbatch CONFIG [frames options, e.g. --overwrite]

#SBATCH --job-name=em-movie-build
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

python -u -m pipeline.movie_build --config "$CONFIG" frames "$@"
ENCODE=$(python -m pipeline.movie_build --config "$CONFIG" encode --print)
# NEMO's FFmpeg module, loaded only after python: its libraries would shadow the conda env's.
ml FFmpeg/7.1.1-GCCcore-14.2.0
eval "$ENCODE"
echo "movie: ${ENCODE##* }"
