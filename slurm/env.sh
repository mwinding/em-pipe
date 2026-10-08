# Cluster environment for em-pipe jobs, sourced by every slurm/*.sbatch and by run_pipeline.sh.
# Defaults are for NEMO at the Crick; set EM_PIPE_ENV to use a different conda env.
#
# One-off setup of the shared env:
#   ml Anaconda3/2024.10 && source /camp/apps/eb/software/Anaconda/conda.env.sh
#   conda env create -f environment.yml -p /camp/lab/windingm/home/shared/conda-envs/em-pipe

ml purge
ml Anaconda3/2024.10
source /camp/apps/eb/software/Anaconda/conda.env.sh
conda activate "${EM_PIPE_ENV:-/camp/lab/windingm/home/shared/conda-envs/em-pipe}"

# Keep numerical libraries to the CPUs Slurm gave us.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
