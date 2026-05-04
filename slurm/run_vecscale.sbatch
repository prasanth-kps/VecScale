#!/bin/bash
#SBATCH --job-name=vecscale_run
#SBATCH --output=slurm/logs/vecscale_%j.out
#SBATCH --error=slurm/logs/vecscale_%j.err
#SBATCH --partition=instruction
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G

set -euo pipefail

cd "$HOME/VecScale"
mkdir -p slurm/logs data results

module purge
# Euler Gen 8: try the 2024-06 software stack first; fall back to older modules.
if module load stack/2024-06 2>/dev/null; then
    module load gcc/13.2.0 cmake/3.27.7 2>/dev/null || module load gcc cmake
else
    module load gcc/14.3.0 cmake/4.1.2
fi

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j "${SLURM_CPUS_PER_TASK}"

# Generate 100k-vector dataset (dim=128, seed=42).
./build/prepare_dataset data/dataset_cpp 100000 1000 128 42

# Run: 4 shards, top-k=10, cpu backend, results → results/.
./build/run_demo data/dataset_cpp 4 10 cpu 0 results
