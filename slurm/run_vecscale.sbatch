#!/bin/bash
#SBATCH --job-name=vecscale_run
#SBATCH --output=vecscale_run-%j.out
#SBATCH --error=vecscale_run-%j.err
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
fi

BACKEND="${VECSCALE_BACKEND:-openmp}"        # cpu|openmp|cuda
OMP_THREADS="${VECSCALE_OMP_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
CMAKE_ARGS="-DVECSCALE_ENABLE_MPI=ON -DVECSCALE_ENABLE_OPENMP=ON"
if [[ "${BACKEND}" == "cuda" ]]; then
  if type module >/dev/null 2>&1; then
    module load gcc/14.3.0 cmake/4.1.0 nvidia/cuda/12.9.1
  fi
  export CC="$(which gcc)"
  export CXX="$(which g++)"
  export CUDAHOSTCXX="$(which g++)"
  CMAKE_ARGS="${CMAKE_ARGS} -DVECSCALE_ENABLE_CUDA=ON -DCMAKE_CUDA_HOST_COMPILER=${CUDAHOSTCXX}"
else
  if type module >/dev/null 2>&1; then
    module load gnu15/15.1.0 cmake/4.1.0
  fi
fi

cmake -S . -B build ${CMAKE_ARGS}
cmake --build build -j "${SLURM_CPUS_PER_TASK:-1}"

./build/prepare_dataset data/dataset_cpp 100000 1000 128 42

if [[ "${VECSCALE_USE_MPI:-1}" -eq 1 && -x ./build/run_demo_mpi ]]; then
  echo "Launching MPI mode with backend=${BACKEND}"
  srun --ntasks="${SLURM_NTASKS:-4}" ./build/run_demo_mpi data/dataset_cpp 10 "${BACKEND}" "${OMP_THREADS}"
else
  echo "Launching single-process mode with backend=${BACKEND}"
  ./build/run_demo data/dataset_cpp 4 10 "${BACKEND}" "${OMP_THREADS}" single
fi
