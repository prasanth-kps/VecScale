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

# ── Step 1: discover what MPI modules exist on this cluster ──────────────────
echo "=== Available MPI-related modules ==="
module spider 2>&1 | grep -i "mpi\|openmpi\|mvapich\|impi" || true
echo "=== (end module list) ==="

# ── Step 2: try to load a toolchain that ships MPI wrappers ──────────────────
# OpenHPC clusters bundle compiler + MPI in a single meta-module like
#   gnu15/15.1.0  +  openmpi5-gnu15/5.0.8
# or provide Intel MPI separately from the Intel compiler.
# We try the most common patterns in order.

LOADED_MPI=""
for CANDIDATE in \
    "openmpi5-gnu15/5.0.8"  \
    "openmpi5-gnu15"         \
    "openmpi4-gnu12/4.1.6"  \
    "openmpi4-gnu12"         \
    "openmpi/5.0.8"          \
    "openmpi/4.1.6"          \
    "openmpi"                \
    "impi/2021.14"           \
    "impi/2025.0"            \
    "intel-mpi/2025.0.4"     \
    "mvapich2-gnu15"         \
    "mvapich2"               \
; do
    if module load "${CANDIDATE}" 2>/dev/null; then
        echo "Loaded MPI module: ${CANDIDATE}"
        LOADED_MPI="${CANDIDATE}"
        break
    fi
done

# Now load the compiler + cmake (use whatever pairs with the MPI above).
if [[ "${LOADED_MPI}" == *gnu15* ]] || [[ "${LOADED_MPI}" == *openmpi5* ]]; then
    module load gnu15/15.1.0 cmake/4.1.2
    echo "Compiler: GNU 15"
elif [[ "${LOADED_MPI}" == *gnu12* ]] || [[ "${LOADED_MPI}" == *openmpi4* ]]; then
    module load gcc/13.2.0 cmake/4.1.2
    echo "Compiler: GCC 13"
else
    # Intel MPI — load intel compiler alongside it
    module load intel/2025.0.4 cmake/4.1.2 2>/dev/null || module load gcc/13.2.0 cmake/4.1.2
    echo "Compiler: Intel or GCC fallback"
fi

# ── Step 3: confirm wrappers are now in PATH ──────────────────────────────────
echo "--- MPI wrapper check ---"
which mpicxx  2>/dev/null && mpicxx  --version 2>&1 | head -1 || echo "mpicxx  NOT found"
which mpicc   2>/dev/null && mpicc   --version 2>&1 | head -1 || echo "mpicc   NOT found"
echo "-------------------------"

if ! command -v mpicxx >/dev/null 2>&1; then
    echo "ERROR: mpicxx not in PATH even after module load."
    echo "Please run 'module spider mpi' on the login node to find the correct module name."
    exit 1
fi

MPI_CXX_BIN=$(which mpicxx)
MPI_C_BIN=$(which mpicc)
echo "Using MPI wrapper: ${MPI_CXX_BIN}"

# ── Step 4: build ─────────────────────────────────────────────────────────────
rm -rf build_mpi
cmake -S . -B build_mpi \
    -DCMAKE_BUILD_TYPE=Release \
    -DVECSCALE_ENABLE_MPI=ON \
    -DMPI_CXX_COMPILER="${MPI_CXX_BIN}" \
    -DMPI_C_COMPILER="${MPI_C_BIN}"
cmake --build build_mpi -j "${SLURM_CPUS_PER_TASK}"

if [ ! -f "build_mpi/run_demo_mpi" ]; then
    echo "ERROR: run_demo_mpi was not produced — MPI not detected by cmake despite wrapper."
    exit 1
fi

# ── Step 5: run ──────────────────────────────────────────────────────────────
build_mpi/prepare_dataset data/dataset_cpp 100000 1000 128 42

echo "===== MPI run: ${SLURM_NTASKS} ranks ====="
srun build_mpi/run_demo_mpi data/dataset_cpp 10 cpu 0 results
