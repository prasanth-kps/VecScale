# VecScale: Distributed GPU ANN Retrieval Prototype for Real-Time Vector Search

**Spring 2026 ME/CS/ECE 759 Final Project Report**
**University of Wisconsin–Madison**

Karthik Reddy Jannupalli · Penchala Siva Prasanth Kasa
May 5, 2026

---

## Abstract

Vector retrieval is foundational to modern AI pipelines — powering retrieval-augmented generation (RAG), semantic search, and recommendation systems. As embedding dimensionality and corpus sizes grow, single-node CPU search becomes a throughput bottleneck. This project, **VecScale**, implements a distributed, GPU-accelerated approximate nearest-neighbor (ANN) retrieval prototype in C++17 that is evaluated rigorously on the UW–Madison Euler HPC cluster.

We proposed to build a system with three parallel execution paths — CPU multi-threading, OpenMP intra-shard parallelism, and CUDA batched matrix search — wrapped in an MPI-based scatter-gather layer for multi-rank distributed execution. We delivered all of these, along with a fully automated benchmark harness that reports bulk throughput (QPS), per-query p50/p95 latency, and Recall@K against an exact brute-force baseline.

Experimental results on Euler confirm perfect linear scaling of the shard-parallel CPU path (1→2→4 shards: 1×→2×→4× QPS), ~4× throughput gain with 4 MPI ranks, and a peak **11.6× GPU speedup** over the single-threaded baseline — all while maintaining **Recall@10 = 1.000** in every run.

**Git repository:** https://github.com/prasanth-kps/VecScale

---

## Table of Contents

1. Problem Statement
2. Solution Description
3. Overview of Results — Demonstration of the Project
4. Deliverables: Building and Running the Project
5. Conclusions and Future Work
6. References

---

## General Information

- **Name:** Karthik Reddy Jannupalli
- **Email:** jannupalli@wisc.edu
- **Home department:** Computer Science
- **Status:** MS Student
- **Teammate:** Penchala Siva Prasanth Kasa (jannupalli@wisc.edu) — this report is identical to their submission, as all code and experiments were developed jointly.
- I release the ME759 Final Project code as open source and under a BSD3 license for unfettered use of it by any interested party.

---

## 1. Problem Statement

### What We Wanted to Accomplish

We set out to build a distributed, GPU-accelerated vector retrieval prototype for approximate nearest-neighbor (ANN) search over high-dimensional embeddings. The system stores a large corpus of floating-point vectors, partitions them into shards, routes incoming query batches to all shards in parallel, and merges the shard-local top-K results into a global answer. The core objective is to measure — quantitatively and reproducibly — how throughput (QPS), query latency (p50/p95), and retrieval quality (Recall@K) change as we scale across:

- Number of CPU shard threads (1, 2, 4, 8)
- OpenMP thread count per shard
- Number of MPI ranks (1, 4)
- Compute backend (serial CPU, OpenMP, CUDA)

### Motivation and Rationale

Vector search is the computational core of every modern retrieval-augmented AI system. Tools like FAISS, Milvus, and Qdrant are production-grade, but their internals are opaque and their distributed layers are complex. We chose this project because it sits at the intersection of three ME759 topics — GPU programming, shared-memory parallelism, and distributed communication — and because a from-scratch implementation in C++17 forces a deep understanding of every design decision:

- **Why does shard parallelism improve throughput?** Because each shard is an independent top-K over a disjoint subset of the corpus; threads never need to communicate until the final merge.
- **Why does CUDA dominate for large batches?** Because cosine similarity over 100K×128-dimensional vectors is a dense matrix-matrix multiply — the ideal GPU workload.
- **What does MPI add?** Memory-distributed scale-out: each rank holds only `corpus_size / num_ranks` vectors, enabling corpora that exceed a single node's RAM.

We deliberately scoped out full consensus/replication protocols (e.g., Raft) and large-scale ingestion pipelines to keep the deliverables measurable and the codebase tractable within one semester.

---

## 2. Solution Description

### Architecture Overview

VecScale is a single-library (`libvecscale.a`) plus two executable entry points:

```
run_demo      — single-node benchmark (CPU threads / OpenMP / CUDA)
run_demo_mpi  — distributed benchmark (MPI scatter-gather across ranks)
```

The system is structured around five components that map one-to-one onto the C++ source files:

```
src/vecscale/
  data.cpp          — CSV dataset load / write (embeddings, queries, IDs)
  worker.cpp         — per-shard search (delegates to compute backend)
  compute.cpp        — CPU serial & OpenMP top-K cosine similarity
  compute_cuda.cu    — CUDA batched cosine similarity kernel
  router.cpp         — scatter queries to workers via std::thread, gather results
  aggregator.cpp     — merge N×top-K shard results into global top-K
  baselines.cpp      — exact brute-force baseline + mean_recall_at_k
  benchmark.cpp      — per-query latency timing harness
```

### Data Structures

```cpp
using Matrix   = std::vector<std::vector<float>>;  // [num_vectors][dim]
using IdArray  = std::vector<std::int64_t>;         // corpus IDs
using TopKResult = std::vector<std::vector<std::int64_t>>;  // [queries][k]

struct GlobalSearchResult {
    TopKResult ids;     // retrieved IDs per query
    Matrix     scores;  // corresponding cosine similarities
};

struct RuntimeConfig {
    Backend backend;    // CPU | OPENMP | CUDA
    int     omp_threads;
};
```

All vectors are stored in row-major `std::vector<std::vector<float>>`. The flat layout was chosen for simplicity and ease of CSV serialisation over raw binary for debuggability on the cluster.

### Component Details

#### 1. Data Layer (`data.cpp`)

Reads and writes three CSV files: `embeddings.csv` (corpus), `queries.csv`, and `ids.csv`. The `prepare_dataset` binary generates synthetic data with configurable size, dimensionality, and random seed. Data is normalised to unit norm before storage to make cosine similarity equivalent to a dot product.

#### 2. Worker and Compute Backends (`worker.cpp`, `compute.cpp`, `compute_cuda.cu`)

Each `Worker` owns a contiguous shard of the corpus. Its `search()` method delegates to one of three backends selected at runtime via `RuntimeConfig::backend`:

- **CPU (serial):** For each query, iterates over all shard vectors and computes dot products. O(Q × N/S × D) time.
- **OpenMP:** Same loop with `#pragma omp parallel for` and a thread-local top-K heap. The number of threads is set via `omp_set_num_threads(config.omp_threads)`.
- **CUDA:** Computes the full similarity matrix `Q × (N/S)` as a batched matrix multiply on the GPU using `cublasSgemm`. Top-K selection is performed with `thrust::partial_sort` on the GPU result buffer. The CUDA path compiles only when `-DVECSCALE_ENABLE_CUDA=ON` is passed to CMake; otherwise `compute_cuda_stub.cpp` provides a no-op that throws if invoked.

#### 3. Query Router (`router.cpp`)

The `QueryRouter` holds a `std::vector<Worker>` (one per shard). Its `search()` method dispatches each worker into a detached `std::thread`, waits for all threads to complete, and then calls the aggregator. No mutex is needed because each thread writes into a pre-sized, non-overlapping slot of `shard_ids[i]` / `shard_scores[i]`.

Key design decision: shard parallelism is implemented with `std::thread` rather than OpenMP so that it is always active regardless of the chosen compute backend. OpenMP threads are reserved for the *within-shard* inner loop, and `std::thread` handles the *across-shard* outer loop. This avoids nested parallelism issues.

#### 4. Aggregator (`aggregator.cpp`)

`merge_topk` takes N vectors of per-shard top-K results and produces a global top-K via a priority queue (min-heap of size K). Time complexity: O(N × K × log K) per query — negligible compared to the search time.

#### 5. Baselines and Recall (`baselines.cpp`)

`exact_baseline_topk` runs a full brute-force search (no sharding, single thread) over the entire corpus. Its result is the ground truth. `mean_recall_at_k` computes per-query recall as the fraction of the exact top-K IDs that appear in the approximate top-K list, then averages across queries:

```
recall@K = (1/Q) * Σ_q  |approx_topK(q) ∩ exact_topK(q)| / K
```

#### 6. MPI Distributed Layer (`run_demo_mpi.cpp`)

The MPI executable uses a classical scatter-gather pattern:

1. **Rank 0** loads the full dataset and splits the corpus into `world_size` equal shards.
2. **`MPI_Scatter`** distributes one shard's embedding matrix to each rank.
3. **Each rank** runs a local top-K search over its shard using the same `Worker` + `compute.cpp` code.
4. **`MPI_Gather`** collects all per-rank results back to rank 0.
5. **Rank 0** merges the gathered candidates through the aggregator, computes the exact baseline, evaluates Recall@K, and writes the benchmark report.

Communication volume per query batch: `world_size × top_K × (sizeof(float) + sizeof(int64_t)) × Q` — small relative to the compute time for this corpus size.

### Build System

CMake 3.16+ with three optional features controlled by `-D` flags:

| Flag | Default | Effect |
|------|---------|--------|
| `VECSCALE_ENABLE_OPENMP` | ON | Links OpenMP if found |
| `VECSCALE_ENABLE_MPI` | ON | Builds `run_demo_mpi` if MPI found |
| `VECSCALE_ENABLE_CUDA` | OFF | Compiles `.cu` kernel, links cuBLAS/Thrust |

`std::thread` is always enabled via `find_package(Threads REQUIRED)`.

### Slurm Automation

Five Slurm batch scripts in `slurm/` cover every experimental axis:

| Script | Purpose |
|--------|---------|
| `run_vecscale.sbatch` | Baseline + 4-shard CPU run, builds shared `build/` |
| `run_scaling.sbatch` | CPU thread and OpenMP sweep (1/2/4/8 shards) |
| `run_array.sbatch` | 3-task Slurm array, 1/2/4 shards in parallel |
| `run_mpi.sbatch` | 4-rank MPI run with `gnu15/15.1.0 + openmpi5/5.0.8` |
| `run_gpu.sbatch` | CUDA build (`gcc/12.2.0 + cuda/12.1.0`) and 1/4-shard GPU run |

---

## 3. Overview of Results — Demonstration of the Project

All experiments were run on the UW–Madison Euler HPC cluster (Euler Generation 8). Dataset: 100,000–120,000 vectors × 128 dimensions, 1,000–1,200 queries, top-10 retrieval.

### 3.1 CPU Shard Scaling (Slurm Array Job)

Each array task used a separate shard count (1, 2, or 4) running on 8 CPUs. Workers ran serially within each shard; `std::thread` provided the cross-shard parallelism.

| Shards | Bulk QPS | p50 Latency | p95 Latency | Speedup vs 1 shard |
|--------|----------|-------------|-------------|---------------------|
| 1 | 144 | 13.8 ms | 13.8 ms | 1.0× |
| 2 | 287 | 7.0 ms | 7.2 ms | **2.0×** |
| 4 | 572 | 3.7 ms | 3.7 ms | **4.0×** |

**Recall@10 = 1.000 in all cases.** The scaling is textbook linear: each additional shard thread takes ownership of an independent 1/S slice of the corpus, so there is zero contention, zero synchronisation overhead between shards, and the total work is trivially divisible.

### 3.2 OpenMP Within-Shard Scaling (`run_scaling.sbatch`)

8 CPUs were distributed as `omp_threads = 8 / shards` to avoid oversubscription. Since total CPU count is fixed, bulk QPS stays flat — what changes is the distribution of time between shard parallelism and per-shard thread parallelism.

| Shards | OMP threads/shard | Bulk QPS | p50 Latency | p95 Latency |
|--------|------------------|----------|-------------|-------------|
| 1 | 8 | 655 | 24.9 ms | 34.5 ms |
| 2 | 4 | 662 | 13.0 ms | 16.4 ms |
| 4 | 2 | 665 | 9.3 ms | 13.4 ms |
| 8 | 1 | 647 | 3.3 ms | 3.3 ms |

Throughput is constant at ~655 QPS because the same 8 CPU-threads do the same total work regardless of how they are grouped. However, the p95 latency decreases sharply as shards increase — the shard-parallel `std::thread` path exposes more parallelism earlier in the pipeline. The high p95 at 1 shard / 8 OpenMP threads (34.5 ms) is due to OpenMP thread-pool startup overhead on the first few queries; subsequent queries are served from warm threads.

### 3.3 GPU Scaling (`run_gpu.sbatch`, GCC 12.2 + CUDA 12.1)

| Shards | Bulk QPS | p50 Latency | p95 Latency | vs CPU Baseline |
|--------|----------|-------------|-------------|-----------------|
| 1 | **1,321** | 20.4 ms | 22.3 ms | **8.5×** |
| 4 | **1,806** | 11.1 ms | 12.5 ms | **11.6×** |

The CUDA backend achieves 8–12× throughput gain over the single-threaded CPU baseline. The key insight is that the cosine similarity computation over 100K vectors × 128 dimensions reduces to a **dense BLAS-3 operation** (SGEMM of shape `[Q × D] × [D × N/S]`) — exactly the regime where GPU memory bandwidth and arithmetic throughput dominate.

The per-query p50 latency (20 ms for 1 shard) is higher than the CPU shard path (3.7 ms for 4 shards) because each individual query incurs GPU kernel launch latency and a PCIe synchronisation point. The GPU advantage is in **batch throughput**: when all 1,000 queries are processed as a single matrix, the kernel runs once and amortises all overhead — hence `bulk_qps` is >1,300.

### 3.4 MPI Distributed Execution (`run_mpi.sbatch`)

| MPI Ranks | Baseline QPS | Distributed QPS | Speedup | Recall@10 |
|-----------|-------------|-----------------|---------|-----------|
| 1 (baseline) | 163.9 | — | — | — |
| 4 | 163.9 | **652.7** | **3.98×** | **1.0000** |

With 4 MPI ranks (`gnu15/15.1.0 + openmpi5/5.0.8` on Euler), each rank searched 25,000 of the 100,000-vector corpus in parallel. Rank 0 gathered all candidates and ran the exact merge. The 3.98× speedup (vs ideal 4×) reflects a small communication and merge overhead (~0.5% efficiency loss) — negligible and consistent with the embarrassingly parallel nature of the scatter-gather pattern.

The MPI run required several infrastructure fixes to reach this result: (1) identifying the correct OpenHPC module (`openmpi5/5.0.8` paired with `gnu15/15.1.0`), (2) passing `-DMPI_CXX_COMPILER=$(which mpicxx)` explicitly to cmake, and (3) switching the launcher from `srun` to `mpirun` because Slurm's PMIx was not available for OpenMPI 5 on this cluster (srun would otherwise start independent singletons instead of a communicating job).

### 3.5 Summary Comparison

| Backend | Config | Bulk QPS | Peak Speedup | Recall@10 |
|---------|--------|----------|--------------|-----------|
| CPU serial (baseline) | 1 thread | ~155 | 1× | — |
| CPU threads | 4 shards | 572 | 3.7× | 1.000 |
| OpenMP | 8 shards | 647 | 4.2× | 1.000 |
| MPI | 4 ranks | 653 | 3.98× | 1.000 |
| CUDA | 1 shard | 1,321 | 8.5× | 1.000 |
| CUDA | 4 shards | 1,806 | **11.6×** | 1.000 |

**Recall@10 = 1.000 across every single experiment.** Because our sharding strategy partitions the corpus by contiguous ID range (no approximation is introduced in the routing layer), the distributed search is in fact exact — the global top-K is guaranteed to be found as long as each shard returns its own exact top-K.

---

## 4. Deliverables: Building and Running the Project

### Repository Structure

```
VecScale/
├── CMakeLists.txt                  — main build definition
├── src/vecscale/                   — library source files
│   ├── aggregator.{hpp,cpp}       — top-K merge
│   ├── baselines.{hpp,cpp}        — exact baseline + recall@k
│   ├── benchmark.{hpp,cpp}        — per-query latency harness
│   ├── compute.{hpp,cpp}          — CPU serial + OpenMP kernel
│   ├── compute_cuda.{hpp,cu}      — CUDA batched kernel
│   ├── compute_cuda_stub.cpp      — no-op stub for non-CUDA builds
│   ├── data.{hpp,cpp}             — CSV I/O
│   ├── router.{hpp,cpp}           — std::thread scatter-gather
│   ├── types.hpp                  — shared data structures
│   └── worker.{hpp,cpp}           — per-shard search
├── scripts/
│   ├── prepare_dataset.cpp        — dataset generator executable
│   ├── run_demo.cpp               — single-node benchmark executable
│   └── run_demo_mpi.cpp           — MPI benchmark executable
├── slurm/
│   ├── run_vecscale.sbatch        — basic single-node job
│   ├── run_scaling.sbatch         — CPU/OpenMP sweep
│   ├── run_array.sbatch           — Slurm array (1/2/4 shards)
│   ├── run_mpi.sbatch             — 4-rank MPI job
│   └── run_gpu.sbatch             — CUDA job
├── results/                       — benchmark reports (auto-generated)
│   ├── benchmark_report.txt
│   ├── benchmark_report_mpi.txt
│   ├── arr_s{1,2,4}/benchmark_report.txt
│   ├── gpu_s{1,4}/benchmark_report.txt
│   └── {s,omp}_s{1,2,4,8}/benchmark_report.txt
└── data/dataset_cpp/              — auto-generated CSV dataset
    ├── embeddings.csv
    ├── queries.csv
    └── ids.csv
```

### Prerequisites

| Component | Version |
|-----------|---------|
| C++ compiler | GCC ≥ 11, GCC ≤ 12 for CUDA builds |
| CMake | ≥ 3.16 |
| OpenMP | Bundled with GCC |
| CUDA toolkit | 12.1 (requires host GCC ≤ 12) |
| MPI | `openmpi5/5.0.8` + `gnu15/15.1.0` (Euler) or any MPI ≥ 3 |

### Build Instructions

#### CPU-only (serial + OpenMP + `std::thread`)
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

#### With CUDA
```bash
cmake -S . -B build_gpu \
    -DCMAKE_BUILD_TYPE=Release \
    -DVECSCALE_ENABLE_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build_gpu -j$(nproc)
```

#### With MPI (Euler: `module load gnu15/15.1.0 openmpi5/5.0.8 cmake/4.1.2`)
```bash
cmake -S . -B build_mpi \
    -DCMAKE_BUILD_TYPE=Release \
    -DVECSCALE_ENABLE_MPI=ON \
    -DMPI_CXX_COMPILER=$(which mpicxx)
cmake --build build_mpi -j$(nproc)
```

### Running the Benchmark

#### Step 1 — Generate dataset
```bash
# build/prepare_dataset <output_dir> [num_vectors] [num_queries] [dim] [seed]
build/prepare_dataset data/dataset_cpp 100000 1000 128 42
```

#### Step 2 — Single-node benchmark
```bash
# build/run_demo <dataset_dir> [shards] [top_k] [backend] [omp_threads] [output_dir]
# Backends: cpu | openmp | cuda

# 4 CPU shards (std::thread)
build/run_demo data/dataset_cpp 4 10 cpu 0 results/cpu_s4

# 4 OpenMP shards, 2 threads each
build/run_demo data/dataset_cpp 4 10 openmp 2 results/omp_s4

# 4 GPU shards (CUDA build required)
build_gpu/run_demo data/dataset_cpp 4 10 cuda 0 results/gpu_s4
```

#### Step 3 — MPI benchmark
```bash
mpirun -n 4 build_mpi/run_demo_mpi data/dataset_cpp 10 cpu 0 results
```

#### Step 4 — Slurm (Euler cluster)
```bash
sbatch slurm/run_vecscale.sbatch   # builds shared binary, 4-shard CPU run
sbatch slurm/run_scaling.sbatch    # CPU + OpenMP sweep
sbatch slurm/run_array.sbatch      # parallel array (reuses build/)
sbatch slurm/run_mpi.sbatch        # 4-rank MPI
sbatch slurm/run_gpu.sbatch        # CUDA build + run
```

All outputs are written to `results/<experiment>/benchmark_report.txt` in a structured key-value format suitable for automated parsing.

---

## 5. Conclusions and Future Work

### What We Accomplished

We fully implemented and benchmarked all three parallelism levels proposed in the project proposal:

1. **Shared-memory shard parallelism** via `std::thread` — achieved perfect linear scaling (4 shards = 4× QPS, 4× latency reduction), confirmed on Euler.
2. **OpenMP within-shard parallelism** — maintained constant throughput on a fixed CPU budget while reducing per-query latency as shard count increases.
3. **CUDA batched distance computation** — achieved an **11.6× speedup** over the CPU baseline using a BLAS-3 matrix multiply formulation of cosine similarity, with Recall@10 = 1.000.
4. **MPI distributed execution** — achieved **3.98× speedup** with 4 ranks using OpenMPI 5 on Euler, with full scatter-gather communication and exact top-K merge.

All four paths were tested end-to-end on Euler with Slurm, and all deliver perfect recall because our sharding strategy introduces no approximation.

### Lessons Learned

- **Shard-parallel architecture is the right abstraction.** Separating the routing layer (`std::thread`) from the compute layer (OpenMP/CUDA) allowed the two to be optimised independently and composed without nested parallelism issues.
- **GPU benefits from batching.** A single 1,000-query batch achieves 1,321 QPS; processing queries one at a time would incur 1,000× the kernel launch overhead. System design must expose batch interfaces to GPU backends.
- **MPI infrastructure on HPC clusters is fragile.** Identifying the correct module pair (`gnu15 + openmpi5`), providing cmake with an explicit `-DMPI_CXX_COMPILER` hint, and switching from `srun` to `mpirun` (due to missing PMIx support) required systematic debugging across six job submissions.
- **Recall@10 = 1.0 is achievable with exact sharding.** When the corpus is partitioned by ID range and each shard returns its exact local top-K, the global merge is exact. Approximation only enters if the sharding introduces geographic bias (e.g., clustering-based partitioning) — a direction for future work.

### How the Project Leveraged ME759 Material

| ME759 Topic | How It Was Applied |
|-------------|-------------------|
| GPU parallel computing | CUDA SGEMM kernel for batched cosine similarity; Thrust partial_sort for top-K selection |
| OpenMP | Parallel inner loop over shard vectors; thread count configured dynamically |
| `std::thread` | Cross-shard parallelism; pre-sized slot assignment eliminates mutex contention |
| MPI | Scatter-gather distributed search; `MPI_Scatter` / `MPI_Gather` for corpus and results |
| Strong scaling | Fixed corpus, increasing shard/rank count — verified linear QPS scaling |
| Performance measurement | Bulk QPS, p50/p95 latency, Recall@K reported for every run |
| HPC cluster (Slurm) | All experiments automated via sbatch; module management, job arrays, parallel builds |

### Future Work

- **Approximate sharding (IVF/HNSW):** Replace exact brute-force per shard with an index structure (Inverted File Index or Hierarchical NSW). This would reduce per-query compute from O(N/S × D) to O(√(N/S) × D) at the cost of some recall, enabling much larger corpora.
- **Multi-node scaling:** Extend the MPI job to use `--nodes=2` or more. The current implementation is already multi-rank but was tested only within a single node due to partition constraints; true multi-node benchmarks would reveal network communication overhead.
- **Dynamic load balancing:** If embeddings cluster unevenly, some shards finish faster than others. A work-stealing or query-to-shard affinity scheme could reduce tail latency.
- **Production hardening:** Add gRPC-based service interface, persistent index (mmap-backed), and incremental ingestion support for a deployable prototype.

---

## References

[1] J. Johnson, M. Douze, and H. Jégou, "Billion-scale similarity search with GPUs," *IEEE Transactions on Big Data*, vol. 7, no. 3, pp. 535–547, 2021. (FAISS)

[2] Y. Malkov and D. Yashunin, "Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 42, no. 4, pp. 824–836, 2020. (HNSW)

[3] T. Hoefler, D. Abts, G. Kumar, S. Scott, and P. Brightwell, "Remote MPI: Towards performance portability of MPI applications," in *IEEE/ACM International Conference on High Performance Computing, Networking, Storage and Analysis (SC)*, 2019.

[4] NVIDIA Corporation, *CUDA C++ Programming Guide*, v12.1, 2023. https://docs.nvidia.com/cuda/cuda-c-programming-guide/

[5] B. Chapman, G. Jost, and R. Van Der Pas, *Using OpenMP: Portable Shared Memory Parallel Programming*. MIT Press, 2008.

[6] Open MPI Project, "Open MPI v5.0.8 Documentation," 2024. https://www.open-mpi.org/doc/v5.0/

[7] UW–Madison Center for High-Throughput Computing, "Euler HPC Cluster User Guide," 2025. https://euler.engr.wisc.edu/
