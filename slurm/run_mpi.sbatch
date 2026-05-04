#!/bin/bash
#SBATCH --job-name=vecscale_mpi
#SBATCH --output=slurm/logs/mpi_%j.out
#SBATCH --error=slurm/logs/mpi_%j.err
#SBATCH --partition=instruction
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

cd "$HOME/VecScale"
mkdir -p slurm/logs data results

module purge
# intel/2025.0.4 bundles Intel MPI (mpicxx in PATH).
# gnu15/15.1.0 is GCC-only and does NOT ship OpenMPI wrappers on this cluster.
if module load intel/2025.0.4 cmake/4.1.2 2>/dev/null; then
    echo "Using Intel compiler + Intel MPI"
elif module load gnu15/15.1.0 cmake/4.1.2 2>/dev/null; then
    echo "Using GNU15 (MPI availability not guaranteed)"
else
    echo "ERROR: Could not load any suitable toolchain."
    exit 1
fi

cmake -S . -B build_mpi -DCMAKE_BUILD_TYPE=Release -DVECSCALE_ENABLE_MPI=ON
cmake --build build_mpi -j "${SLURM_CPUS_PER_TASK}"

if [ ! -f "build_mpi/run_demo_mpi" ]; then
    echo "ERROR: run_demo_mpi was not built — MPI was not found by cmake."
    echo "Check that mpicxx is in PATH after module load."
    exit 1
fi

build_mpi/prepare_dataset data/dataset_cpp 100000 1000 128 42

echo "===== MPI run: ${SLURM_NTASKS} ranks ====="
srun build_mpi/run_demo_mpi data/dataset_cpp 10 cpu 0 results
