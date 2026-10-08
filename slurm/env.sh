# Cluster environment for em-pipe jobs, sourced by every slurm/*.sbatch and by run_pipeline.sh.
# Defaults are for NEMO at the Crick; set EM_PIPE_ENV to use a different conda env.
#
# One-off setup of the shared env:
#   ml Anaconda3/2024.10
#   conda env create -f environment.yml -p /camp/lab/windingm/home/shared/conda-envs/em-pipe
#
# Conda is activated with `conda shell.bash hook`, which only affects this shell. Don't use the
# Crick conda.env.sh here: it rewrites ~/.bashrc every time it runs, and array tasks starting
# together would race on that file.

# Non-login shells (e.g. `ssh host cmd`) don't have Lmod's `ml` yet.
if ! type ml >/dev/null 2>&1; then
    for f in /etc/profile.d/z00_lmod.sh /etc/profile.d/modules.sh; do
        [ -f "$f" ] && source "$f" && break
    done
fi
ml purge
ml Anaconda3/2024.10
eval "$(conda shell.bash hook)"
conda activate "${EM_PIPE_ENV:-/camp/lab/windingm/home/shared/conda-envs/em-pipe}"

# Keep numerical libraries to the CPUs Slurm gave us.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
