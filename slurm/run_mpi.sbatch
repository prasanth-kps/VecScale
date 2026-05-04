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
# gnu15/15.1.0 is the OpenHPC GCC-15 + OpenMPI 5 toolchain.
# Fall back to intel/2025.0.4 which bundles Intel MPI.
if module load gnu15/15.1.0 cmake/4.1.2 2>/dev/null; then
    echo "Using GNU15 + OpenMPI toolchain"
elif module load intel/2025.0.4 cmake/4.1.2 2>/dev/null; then
    echo "Using Intel MPI toolchain"
else
    echo "ERROR: Could not load any MPI-capable toolchain."
    exit 1
fi

# Use a dedicated build directory so this job never conflicts with other running jobs.
BUILD_DIR="build_mpi"
cmake -S . -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release -DVECSCALE_ENABLE_MPI=ON
cmake --build "${BUILD_DIR}" -j "${SLURM_CPUS_PER_TASK}"

"${BUILD_DIR}/prepare_dataset" data/dataset_cpp 100000 1000 128 42

echo "===== MPI run: ${SLURM_NTASKS} ranks ====="
srun "${BUILD_DIR}/run_demo_mpi" data/dataset_cpp 10 cpu 0 results
