#!/bin/bash
#SBATCH --job-name=vecscale_arr
#SBATCH --array=0-2
#SBATCH --output=slurm/logs/arr_%A_%a.out
#SBATCH --error=slurm/logs/arr_%A_%a.err
#SBATCH --partition=instruction
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G

set -euo pipefail

cd "$HOME/VecScale"
mkdir -p slurm/logs data results

module purge
module load gcc/13.2.0 cmake/4.1.2

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j "${SLURM_CPUS_PER_TASK}"

./build/prepare_dataset data/dataset_cpp 120000 1200 128 42

SHARDS_LIST=(1 2 4)
SHARDS=${SHARDS_LIST[$SLURM_ARRAY_TASK_ID]}

echo "Running shards=$SHARDS"
./build/run_demo data/dataset_cpp "$SHARDS" 10 cpu 0 "results/arr_s${SHARDS}"
