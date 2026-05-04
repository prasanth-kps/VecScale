#!/bin/bash
#SBATCH --job-name=vecscale_scale
#SBATCH --output=slurm/logs/scale_%j.out
#SBATCH --error=slurm/logs/scale_%j.err
#SBATCH --partition=instruction
#SBATCH --time=00:25:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G

set -euo pipefail

cd "$HOME/VecScale"
mkdir -p slurm/logs data results

module purge
if module load stack/2024-06 2>/dev/null; then
    module load gcc/13.2.0 cmake/3.27.7 2>/dev/null || module load gcc cmake
else
    module load gcc/14.3.0 cmake/4.1.2
fi

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j "${SLURM_CPUS_PER_TASK}"

./build/prepare_dataset data/dataset_cpp 120000 1200 128 42

echo "===== scaling run ====="
for s in 1 2 4 8; do
  echo "---- shards=$s ----"
  ./build/run_demo data/dataset_cpp "$s" 10 results
done
