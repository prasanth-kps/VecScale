#!/bin/bash
#SBATCH --job-name=vecscale_mpi
#SBATCH --output=slurm/logs/mpi_%j.out
#SBATCH --error=slurm/logs/mpi_%j.err
#SBATCH --partition=instruction
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

cd "$HOME/VecScale"
mkdir -p slurm/logs data results

module purge
if module load stack/2024-06 2>/dev/null; then
    module load gcc/13.2.0 cmake/3.27.7 openmpi 2>/dev/null || module load gcc cmake openmpi
else
    module load gcc/14.3.0 cmake/4.1.2 openmpi
fi

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DVECSCALE_ENABLE_MPI=ON
cmake --build build -j "${SLURM_CPUS_PER_TASK}"

./build/prepare_dataset data/dataset_cpp 100000 1000 128 42

echo "===== MPI run: ${SLURM_NTASKS} ranks ====="
# Use srun (Slurm-native MPI launcher) — preferred over mpirun on Euler.
srun ./build/run_demo_mpi data/dataset_cpp 10 cpu 0 results
