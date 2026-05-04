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
# intel/2025.0.4 bundles Intel MPI (mpicxx/mpiicpx in PATH).
# gnu15/15.1.0 is GCC-only — no OpenMPI wrappers on this cluster.
if module load intel/2025.0.4 cmake/4.1.2 2>/dev/null; then
    echo "Using Intel compiler + Intel MPI"
elif module load gnu15/15.1.0 cmake/4.1.2 2>/dev/null; then
    echo "Using GNU15 (MPI availability not guaranteed)"
else
    echo "ERROR: Could not load any suitable toolchain."
    exit 1
fi

# Diagnose MPI wrapper visibility so we know what cmake will find.
echo "--- MPI wrapper check ---"
which mpicxx   2>/dev/null && mpicxx   --version 2>&1 | head -1 || echo "mpicxx   NOT found"
which mpiicpx  2>/dev/null && mpiicpx  --version 2>&1 | head -1 || echo "mpiicpx  NOT found"
which mpicc    2>/dev/null && mpicc    --version 2>&1 | head -1 || echo "mpicc    NOT found"
echo "-------------------------"

# Resolve the C++ MPI wrapper; prefer the generic 'mpicxx', fall back to the
# Intel oneAPI-specific 'mpiicpx'.
if   command -v mpicxx  >/dev/null 2>&1; then MPI_CXX_BIN=$(which mpicxx)
elif command -v mpiicpx >/dev/null 2>&1; then MPI_CXX_BIN=$(which mpiicpx)
else
    echo "ERROR: No MPI C++ wrapper found in PATH. Cannot build run_demo_mpi."
    exit 1
fi
echo "Using MPI wrapper: ${MPI_CXX_BIN}"

# Pass the wrapper path explicitly so find_package(MPI) cannot miss it.
rm -rf build_mpi
cmake -S . -B build_mpi \
    -DCMAKE_BUILD_TYPE=Release \
    -DVECSCALE_ENABLE_MPI=ON \
    -DMPI_CXX_COMPILER="${MPI_CXX_BIN}" \
    -DMPI_C_COMPILER="$(which mpicc 2>/dev/null || echo mpicc)"
cmake --build build_mpi -j "${SLURM_CPUS_PER_TASK}"

if [ ! -f "build_mpi/run_demo_mpi" ]; then
    echo "ERROR: run_demo_mpi was not built — MPI was not found by cmake."
    echo "mpicxx used: ${MPI_CXX_BIN}"
    exit 1
fi

build_mpi/prepare_dataset data/dataset_cpp 100000 1000 128 42

echo "===== MPI run: ${SLURM_NTASKS} ranks ====="
srun build_mpi/run_demo_mpi data/dataset_cpp 10 cpu 0 results
