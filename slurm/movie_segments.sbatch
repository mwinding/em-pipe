#!/bin/bash
# The mesh layers of a tissue movie's segments (tissue_movie.segments; Blender, EM_PIPE_BLENDER in
# slurm/site.sh), on their own: on a GPU node they render many times faster than on CPUs. Then
# slurm/movie_tissue.sbatch with the same options (its meshes step finds them done). With --frames, only
# those frames' layers, and then those frames too (a quick look at a few stills).
# Usage: sbatch --partition=GPU_PARTITION --gres=gpu:1 --output=LOG slurm/movie_segments.sbatch CONFIG [--preview N] [--frames 600,700]

#SBATCH --job-name=em-movie-segments
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

python -u -m pipeline.movie_tissue --config "$CONFIG" meshes "$@"
case " $* " in *" --frames "*) python -u -m pipeline.movie_tissue --config "$CONFIG" frames "$@" ;; esac
