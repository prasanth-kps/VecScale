#!/bin/bash
#SBATCH --job-name=vecscale_gpu
#SBATCH --output=slurm/logs/gpu_%j.out
#SBATCH --error=slurm/logs/gpu_%j.err
#SBATCH --partition=gpu.sm_86          # adjust to the GPU partition on your Euler cluster
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

set -euo pipefail

cd "$HOME/VecScale"
mkdir -p slurm/logs data results

module purge
# Load a GCC + CUDA toolchain compatible with Euler Gen 8.
if module load stack/2024-06 2>/dev/null; then
    module load gcc/13.2.0 cmake/3.27.7 cuda/12.1.1 2>/dev/null || \
    module load gcc cmake cuda
else
    module load gcc/14.3.0 cmake/4.1.2 cuda
fi

# Build with CUDA backend enabled.
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DVECSCALE_ENABLE_CUDA=ON

cmake --build build -j "${SLURM_CPUS_PER_TASK}"

./build/prepare_dataset data/dataset_cpp 100000 1000 128 42

echo "===== GPU backend (1 shard — full corpus on single GPU) ====="
./build/run_demo data/dataset_cpp 1 10 cuda 0 results/gpu_s1

echo "===== GPU backend (4 shards — each shard on its own thread, GPU kernel per shard) ====="
./build/run_demo data/dataset_cpp 4 10 cuda 0 results/gpu_s4
