#!/bin/bash
#SBATCH --job-name=vecscale_gpu
#SBATCH --output=slurm/logs/gpu_%j.out
#SBATCH --error=slurm/logs/gpu_%j.err
#SBATCH --partition=instruction
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

set -euo pipefail

cd "$HOME/VecScale"
mkdir -p slurm/logs data results

module purge
# CUDA 12.1 requires GCC <= 12. gcc/12.2.0 is the latest GCC on this
# cluster that satisfies that constraint.
module load gcc/12.2.0 cmake/4.1.2 nvidia/cuda/12.1.0

cmake -S . -B build_gpu \
    -DCMAKE_BUILD_TYPE=Release \
    -DVECSCALE_ENABLE_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=86

cmake --build build_gpu -j "${SLURM_CPUS_PER_TASK}"

build_gpu/prepare_dataset data/dataset_cpp 100000 1000 128 42

echo "===== GPU backend: 1 shard ====="
build_gpu/run_demo data/dataset_cpp 1 10 cuda 0 results/gpu_s1

echo "===== GPU backend: 4 shards ====="
build_gpu/run_demo data/dataset_cpp 4 10 cuda 0 results/gpu_s4
