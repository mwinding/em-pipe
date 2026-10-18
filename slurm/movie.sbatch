#!/bin/bash
# Fly-through movie of a rendered volume -> output_dir/<movie name>.mp4 (settings: the config's movie:)
# Usage: sbatch --output=LOG slurm/movie.sbatch CONFIG [frames options, e.g. --overwrite]

#SBATCH --job-name=em-movie
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

python -u -m pipeline.movie --config "$CONFIG" frames "$@"
ENCODE=$(python -m pipeline.movie --config "$CONFIG" encode --print)
# The site's FFmpeg module (slurm/site.sh), loaded only after python: its libraries could shadow the env's.
[ -z "${EM_PIPE_FFMPEG_MODULE:-}" ] || ml "$EM_PIPE_FFMPEG_MODULE"
eval "$ENCODE"
echo "movie: ${ENCODE##* }"
