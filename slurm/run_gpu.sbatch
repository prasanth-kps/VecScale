#!/bin/bash
#SBATCH --job-name=vecscale_gpu
#SBATCH --output=slurm/logs/gpu_%j.out
#SBATCH --error=slurm/logs/gpu_%j.err
#SBATCH --partition=instruction
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

set -euo pipefail

cd "$HOME/VecScale"
mkdir -p slurm/logs data results

module purge
# gcc/13.2.0 + nvidia/cuda/12.1.0 are both confirmed available on this cluster.
module load gcc/13.2.0 cmake/4.1.2 nvidia/cuda/12.1.0

cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DVECSCALE_ENABLE_CUDA=ON

cmake --build build -j "${SLURM_CPUS_PER_TASK}"

./build/prepare_dataset data/dataset_cpp 100000 1000 128 42

echo "===== GPU backend: 1 shard (full corpus, single GPU) ====="
./build/run_demo data/dataset_cpp 1 10 cuda 0 results/gpu_s1

echo "===== GPU backend: 4 shards (each shard dispatched in its own thread) ====="
./build/run_demo data/dataset_cpp 4 10 cuda 0 results/gpu_s4
