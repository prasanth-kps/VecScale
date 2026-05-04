#!/bin/bash
#SBATCH --job-name=vecscale_arr
#SBATCH --array=0-2
#SBATCH --output=slurm/logs/arr_%A_%a.out
#SBATCH --error=slurm/logs/arr_%A_%a.err
#SBATCH --partition=instruction
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G

set -euo pipefail

cd "$HOME/VecScale"
mkdir -p slurm/logs data results

module purge
module load gcc/13.2.0 cmake/4.1.2

# Reuse the shared build/ if run_scaling or run_vecscale already built it.
# Only build from scratch if the binary is missing.
if [ -f "build/run_demo" ]; then
    BUILD_DIR="build"
    echo "Reusing existing build in build/"
else
    # Build in a task-specific directory to avoid simultaneous-build races.
    BUILD_DIR="build_task_${SLURM_ARRAY_TASK_ID}"
    cmake -S . -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
    cmake --build "${BUILD_DIR}" -j "${SLURM_CPUS_PER_TASK}"
fi

"${BUILD_DIR}/prepare_dataset" data/dataset_cpp 120000 1200 128 42

SHARDS_LIST=(1 2 4)
SHARDS=${SHARDS_LIST[$SLURM_ARRAY_TASK_ID]}

echo "Running shards=$SHARDS"
"${BUILD_DIR}/run_demo" data/dataset_cpp "$SHARDS" 10 cpu 0 "results/arr_s${SHARDS}"
