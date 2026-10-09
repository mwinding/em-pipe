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

# Non-login shells (e.g. `ssh host cmd`) have neither Lmod's `ml` nor the site's MODULEPATH
# (NEMO sets it in /etc/profile.d/00-modulepath.sh): do the login setup. /etc/profile doesn't read
# ~/.bashrc. Its scripts aren't written for `set -e`, so that is paused meanwhile.
if ! type ml >/dev/null 2>&1; then
    case $- in *e*) _em_e=1 ;; *) _em_e= ;; esac
    set +e
    source /etc/profile >/dev/null 2>&1
    [ -n "$_em_e" ] && set -e
    unset _em_e
fi
ml purge
ml Anaconda3/2024.10
EM_PIPE_ENV=${EM_PIPE_ENV:-/camp/lab/windingm/home/shared/conda-envs/em-pipe}
eval "$(conda shell.bash hook)"
conda activate "$EM_PIPE_ENV"
# A job inherits the submitting shell's activation, so conda treats the activate above as already
# done although `ml` has just put the base Anaconda first on PATH: make sure the env comes first.
case ":$PATH:" in
    ":$EM_PIPE_ENV/bin:"*) ;;
    *) export PATH="$EM_PIPE_ENV/bin:$PATH" ;;
esac

# Keep numerical libraries to the CPUs Slurm gave us.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
