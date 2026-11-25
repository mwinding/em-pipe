#!/bin/bash
# Z-relations: each tile against itself in the next slices, from the preview thumbnails (run: array over
# z-chunks), then merge -> output_dir/work/zrelations/relations.csv (for intensity.balance_z_distance > 0)
# Usage: sbatch slurm/zrelations.sbatch CONFIG run|merge [extra args]   (normally submitted by run_pipeline.sh)
# Thumbnails only: ~0.1 s per tile slice.

#SBATCH --job-name=em-zrelations
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mail-type=FAIL

set -eo pipefail
CONFIG=$1; shift
cd "${EM_PIPE_ROOT:-$SLURM_SUBMIT_DIR}"
source slurm/env.sh

python -u -m pipeline.zrelations --config "$CONFIG" "$@"
