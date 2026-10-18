#!/bin/bash
# Whole-volume movie (the block circling, the resin turning transparent, only the tissue left; with
# tissue_movie.fly_through, then the fly-through) -> output_dir/<name>.mp4 (settings: the config's
# tissue_movie: and, for fly_through, movie:). One frame per process: --workers below.
# Usage: sbatch --output=LOG slurm/movie_tissue.sbatch CONFIG [frames options, e.g. --overwrite]

#SBATCH --job-name=em-movie-tissue
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mail-type=FAIL

set -eo pipefail
CONFIG=$1; shift
cd "${EM_PIPE_ROOT:-$SLURM_SUBMIT_DIR}"
source slurm/env.sh

python -u -m pipeline.movie_tissue --config "$CONFIG" frames --workers "${SLURM_CPUS_PER_TASK:-8}" "$@"
ENCODE=$(python -m pipeline.movie_tissue --config "$CONFIG" encode --print)
# The site's FFmpeg module (slurm/site.sh), loaded only after python: its libraries could shadow the env's.
[ -z "${EM_PIPE_FFMPEG_MODULE:-}" ] || ml "$EM_PIPE_FFMPEG_MODULE"
eval "$ENCODE"
echo "movie: ${ENCODE##* }"
