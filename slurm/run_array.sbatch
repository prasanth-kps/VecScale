#!/bin/bash
#SBATCH --job-name=vecscale_arr
#SBATCH --array=0-2
#SBATCH --output=logs/arr_%A_%a.out
#SBATCH --error=logs/arr_%A_%a.err
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

./build/prepare_dataset data/dataset_cpp 120000 1200 128 42

SHARDS_LIST=(1 2 4)
SHARDS=${SHARDS_LIST[$SLURM_ARRAY_TASK_ID]}

echo "Running shards=$SHARDS"
./build/run_demo data/dataset_cpp "$SHARDS" 10
