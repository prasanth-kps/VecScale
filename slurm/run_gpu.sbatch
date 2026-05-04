#!/bin/bash
#SBATCH --job-name=vecscale_gpu
#SBATCH --output=slurm/logs/gpu_%j.out
#SBATCH --error=slurm/logs/gpu_%j.err
#SBATCH --partition=instruction   # change to a GPU partition if your account has access
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

set -euo pipefail

cd "$HOME/VecScale"
mkdir -p slurm/logs data results

module purge
# gcc/13.2.0 + nvidia/cuda/12.1.0 are confirmed available on this cluster.
module load gcc/13.2.0 cmake/4.1.2 nvidia/cuda/12.1.0

BUILD_DIR="build_gpu"
cmake -S . -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DVECSCALE_ENABLE_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=86

cmake --build "${BUILD_DIR}" -j "${SLURM_CPUS_PER_TASK}"

"${BUILD_DIR}/prepare_dataset" data/dataset_cpp 100000 1000 128 42

echo "===== GPU backend: 1 shard ====="
"${BUILD_DIR}/run_demo" data/dataset_cpp 1 10 cuda 0 results/gpu_s1

echo "===== GPU backend: 4 shards ====="
"${BUILD_DIR}/run_demo" data/dataset_cpp 4 10 cuda 0 results/gpu_s4
