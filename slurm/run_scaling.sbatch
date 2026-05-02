#!/bin/bash
#SBATCH --job-name=vecscale_scale
#SBATCH --output=logs/scale_%j.out
#SBATCH --error=logs/scale_%j.err
#SBATCH --time=00:25:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G

set -euo pipefail

cd "$HOME/vecscale_test"
mkdir -p logs data results

module purge
module load gcc cmake

cmake -S . -B build
cmake --build build -j "${SLURM_CPUS_PER_TASK}"

./build/prepare_dataset data/dataset_cpp 120000 1200 128 42

echo "===== scaling run ====="
for s in 1 2 4; do
  echo "---- shards=$s ----"
  ./build/run_demo data/dataset_cpp "$s" 10
done
