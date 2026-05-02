#!/bin/bash
#SBATCH --job-name=vecscale_run
#SBATCH --output=logs/vecscale_%j.out
#SBATCH --error=logs/vecscale_%j.err
#SBATCH --partition=instruction
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G

set -euo pipefail

cd "$HOME/vecscale_test"
mkdir -p logs data results

module purge
module load gcc cmake

cmake -S . -B build
cmake --build build -j "${SLURM_CPUS_PER_TASK}"

./build/prepare_dataset data/dataset_cpp 100000 1000 128 42
./build/run_demo data/dataset_cpp 4 10
