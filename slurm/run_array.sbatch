#!/bin/bash
#SBATCH --job-name=vecscale_arr
#SBATCH --array=0-2
#SBATCH --output=vecscale_arr-%A_%a.out
#SBATCH --error=vecscale_arr-%A_%a.err
#SBATCH --ntasks=4
#SBATCH --partition=instruction
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --hint=nomultithread
# Optional for GPU partition runs:
#SBATCH --gpus=1

set -euo pipefail

cd "$HOME/VecScale"
mkdir -p data results

if type module >/dev/null 2>&1; then
  module purge || true
  module load gnu15/15.1.0 cmake/4.1.0
fi

BACKEND="${VECSCALE_BACKEND:-openmp}"        # cpu|openmp|cuda
OMP_THREADS="${VECSCALE_OMP_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
CMAKE_ARGS="-DVECSCALE_ENABLE_MPI=ON -DVECSCALE_ENABLE_OPENMP=ON"
if [[ "${BACKEND}" == "cuda" ]]; then
  if type module >/dev/null 2>&1; then
    module load nvidia/cuda/12.9.1 || true
  fi
  CMAKE_ARGS="${CMAKE_ARGS} -DVECSCALE_ENABLE_CUDA=ON"
fi

cmake -S . -B build ${CMAKE_ARGS}
cmake --build build -j "${SLURM_CPUS_PER_TASK:-1}"

./build/prepare_dataset data/dataset_cpp 120000 1200 128 42

SHARDS_LIST=(1 2 4)
SHARDS=${SHARDS_LIST[$SLURM_ARRAY_TASK_ID]}

echo "Running shards=$SHARDS"
if [[ "${VECSCALE_USE_MPI:-0}" -eq 1 && -x ./build/run_demo_mpi ]]; then
  srun --ntasks="$SHARDS" ./build/run_demo_mpi data/dataset_cpp 10 "${BACKEND}" "${OMP_THREADS}"
else
  ./build/run_demo data/dataset_cpp "$SHARDS" 10 "${BACKEND}" "${OMP_THREADS}" single
fi
