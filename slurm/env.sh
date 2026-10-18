# Cluster environment for em-pipe jobs, sourced by every slurm/*.sbatch and by run_pipeline.sh.
# The site's settings come from slurm/site.sh (git-ignored): copy slurm/site.example.sh there and
# fill it in. It sets EM_PIPE_ENV (the conda env) and optionally EM_PIPE_MODULES (modules to load
# first), EM_PIPE_FFMPEG_MODULE (for the movies) and SBATCH_PARTITION.
#
# Conda is activated with `conda shell.bash hook`, which only affects this shell. Don't call a
# site's conda setup script here if it edits ~/.bashrc (some do, every time they run): array tasks
# starting together would race on that file.

_em_site="$(dirname "${BASH_SOURCE[0]}")/site.sh"
if [ -f "$_em_site" ]; then
    source "$_em_site"
fi
unset _em_site
if [ -z "${EM_PIPE_ENV:-}" ]; then
    echo "slurm/env.sh: EM_PIPE_ENV (the em-pipe conda env) is not set: copy slurm/site.example.sh to slurm/site.sh and fill it in" >&2
    return 1
fi

if [ -n "${EM_PIPE_MODULES:-}" ]; then
    # Non-login shells (e.g. `ssh host cmd`) have neither Lmod's `ml` nor the site's MODULEPATH
    # (set by /etc/profile.d scripts): do the login setup. /etc/profile doesn't read ~/.bashrc. Its
    # scripts aren't written for `set -e`, so that is paused meanwhile.
    if ! type ml >/dev/null 2>&1; then
        case $- in *e*) _em_e=1 ;; *) _em_e= ;; esac
        set +e
        source /etc/profile >/dev/null 2>&1
        [ -n "$_em_e" ] && set -e
        unset _em_e
    fi
    ml purge
    ml $EM_PIPE_MODULES
fi
eval "$(conda shell.bash hook)"
conda activate "$EM_PIPE_ENV"
# A job inherits the submitting shell's activation, so conda treats the activate above as already
# done although `ml` has just put the base conda first on PATH: make sure the env comes first.
case ":$PATH:" in
    ":$EM_PIPE_ENV/bin:"*) ;;
    *) export PATH="$EM_PIPE_ENV/bin:$PATH" ;;
esac

# Keep numerical libraries to the CPUs Slurm gave us.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
