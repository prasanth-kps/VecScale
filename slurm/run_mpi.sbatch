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

# Confirmed available MPI modules on Euler (from 'module spider mpi'):
#   openmpi5/5.0.8   — OpenMPI 5, pairs with gnu15/15.1.0
#   impi/2021.14     — Intel MPI
#   mpich/3.4.3-ucx  — MPICH fallback
#
# Try them in preference order.

if module load gnu15/15.1.0 openmpi5/5.0.8 cmake/4.1.2 2>/dev/null; then
    echo "Toolchain: gnu15/15.1.0 + openmpi5/5.0.8"
elif module load intel/2025.0.4 impi/2021.14 cmake/4.1.2 2>/dev/null; then
    echo "Toolchain: intel/2025.0.4 + impi/2021.14"
elif module load gcc/13.2.0 mpich/3.4.3-ucx cmake/4.1.2 2>/dev/null; then
    echo "Toolchain: gcc/13.2.0 + mpich/3.4.3-ucx"
else
    echo "ERROR: Could not load any MPI toolchain."
    exit 1
fi

# Confirm mpicxx is in PATH.
echo "--- MPI wrapper check ---"
which mpicxx 2>/dev/null && mpicxx --version 2>&1 | head -1 || echo "mpicxx NOT found"
echo "-------------------------"

if ! command -v mpicxx >/dev/null 2>&1; then
    echo "ERROR: mpicxx still not in PATH after module load."
    exit 1
fi

MPI_CXX_BIN=$(which mpicxx)
MPI_C_BIN=$(which mpicc)
echo "Using: ${MPI_CXX_BIN}"

# Build with explicit MPI wrapper hints so cmake cannot miss them.
rm -rf build_mpi
cmake -S . -B build_mpi \
    -DCMAKE_BUILD_TYPE=Release \
    -DVECSCALE_ENABLE_MPI=ON \
    -DMPI_CXX_COMPILER="${MPI_CXX_BIN}" \
    -DMPI_C_COMPILER="${MPI_C_BIN}"
cmake --build build_mpi -j "${SLURM_CPUS_PER_TASK}"

if [ ! -f "build_mpi/run_demo_mpi" ]; then
    echo "ERROR: run_demo_mpi not produced — cmake MPI detection failed."
    exit 1
fi

build_mpi/prepare_dataset data/dataset_cpp 100000 1000 128 42

echo "===== MPI run: ${SLURM_NTASKS} ranks ====="
srun build_mpi/run_demo_mpi data/dataset_cpp 10 cpu 0 results
