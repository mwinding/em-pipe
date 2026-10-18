# Site settings for em-pipe's Slurm jobs. Copy to slurm/site.sh (git-ignored, so each clone keeps
# its own; a symlink to a file shared by your group works too) and fill in. Sourced by
# slurm/env.sh in every job and by run_pipeline.sh.

# The conda env made from environment.yml: conda env create -f environment.yml -p PATH
EM_PIPE_ENV=${EM_PIPE_ENV:-/path/to/conda-envs/em-pipe}

# Environment modules loaded (after `ml purge`) before the env is activated, e.g. the one that
# provides conda. Leave empty if conda is on PATH already.
EM_PIPE_MODULES=""

# FFmpeg for the movies, if it comes from a module. The movie jobs load it after python, as its
# libraries could shadow the env's.
EM_PIPE_FFMPEG_MODULE=""

# The partition run_pipeline.sh submits to (a config's slurm: section can still set one per job).
# Unset: the cluster's default partition.
# export SBATCH_PARTITION=cpu
