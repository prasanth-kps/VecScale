# VecScale

Distributed GPU ANN Retrieval Prototype for Real-Time Vector Search.

VecScale is a C++17 prototype that demonstrates distributed approximate nearest-neighbour (ANN) search over large embedding corpora. The corpus is partitioned into shards, each shard is searched in parallel by a dedicated worker thread, and results are merged into a global top-k ranking. A single-node brute-force baseline is included for recall and throughput comparison.

---

## Requirements

| Tool | Minimum version |
|------|----------------|
| CMake | 3.20 |
| GCC or Clang | GCC 11 / Clang 14 (C++17) |
| OpenMP | any recent version |

On first CMake configure, **Eigen 3.4** and **nlohmann/json 3.11** are downloaded automatically via `FetchContent` (internet required). If they are already installed system-wide, the local copies are used instead.

```bash
# macOS
brew install cmake gcc libomp

# Ubuntu / Debian
sudo apt install cmake build-essential libomp-dev
```

---

## Project Structure

```
VecScale/
├── CMakeLists.txt
├── configs/
│   └── default.json          # runtime configuration
├── include/vecscale/
│   ├── common.hpp            # shared types (Matrix, TopKResult, ShardResult)
│   ├── compute.hpp           # topk_cosine_similarity
│   ├── data.hpp              # Dataset struct + save/load (.vsd format)
│   ├── aggregator.hpp        # merge_topk
│   ├── router.hpp            # QueryRouter class
│   ├── worker.hpp            # Worker class + run_worker_server
│   ├── baselines.hpp         # measure_single_node_baseline
│   └── benchmark.hpp         # BenchmarkConfig, BenchmarkResult, run_benchmark
├── src/
│   ├── compute.cpp           # Eigen + OpenMP cosine similarity + top-k
│   ├── data.cpp              # binary .vsd file I/O
│   ├── aggregator.cpp        # multi-shard result merging
│   ├── router.cpp            # corpus partitioning
│   ├── worker.cpp            # per-shard query processing
│   ├── baselines.cpp         # single-node brute-force reference
│   └── benchmark.cpp         # full pipeline orchestration + report
└── tools/
    ├── prepare_dataset.cpp   # CLI: generate synthetic dataset
    └── run_demo.cpp          # CLI: load config and run benchmark
```

---

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)                        # Linux
make -j$(sysctl -n hw.logicalcpu)      # macOS
```

This produces two executables inside `build/`: `prepare_dataset` and `run_demo`.

---

## Usage

### 1. Generate a dataset

```bash
./build/prepare_dataset \
    --output data/dataset.vsd \
    --num-vectors 100000 \
    --num-queries 1000 \
    --dim 128 \
    --seed 42
```

| Flag | Default | Description |
|------|---------|-------------|
| `--output` | *(required)* | Output path for the `.vsd` binary file |
| `--num-vectors` | 100000 | Number of corpus embedding vectors |
| `--num-queries` | 1000 | Number of query vectors |
| `--dim` | 128 | Embedding dimension |
| `--seed` | 42 | Random seed for reproducibility |

### 2. Run the benchmark

```bash
./build/run_demo --config configs/default.json
```

`default.json` controls all runtime parameters:

```json
{
  "dataset_path": "data/dataset.vsd",
  "output_dir":   "results",
  "num_shards":   4,
  "top_k":        10,
  "batch_size":   8
}
```

The benchmark runs three stages and prints a summary:

```
=== VecScale Distributed Benchmark ===
Loading dataset: data/dataset.vsd
  corpus=100000  queries=1000  dim=128  shards=4  k=10  batch_size=8

[1/3] Single-node baseline (OpenMP)...
  elapsed=Xms  QPS=XXXX

[2/3] Distributed search (4 shards)...
  shard 0: 25000 vectors  ...
  p50=Xms  p99=Xms

[3/3] Computing recall@10...
  recall@10 = 1.0000

Report written to: results/benchmark_report.txt
```

---

## Architecture

```
QueryRouter
  │  splits corpus into N contiguous shards
  │
  ├── Worker 0  ──┐
  ├── Worker 1  ──┤  each runs topk_cosine_similarity on its shard
  ├── Worker 2  ──┤  in a dedicated std::thread
  └── Worker N  ──┘
                  │
             merge_topk()
                  │
           global top-k result
                  │
         compare vs. baseline → recall@k
```

Each worker runs `topk_cosine_similarity` (L2-normalise → matrix multiply → partial sort) with OpenMP disabled so the outer thread pool provides concurrency. The single-node baseline runs the same kernel with OpenMP enabled across all queries for maximum single-machine throughput.

---

## Dataset Format (.vsd)

VecScale uses a custom binary format designed for fast sequential I/O with no external dependencies.

| Offset | Size | Field |
|--------|------|-------|
| 0 | 8 bytes | Magic: `VECSCALE` |
| 8 | 4 bytes | Version (`uint32`, currently `1`) |
| 12 | 8 bytes | `num_vectors` (`int64`) |
| 20 | 8 bytes | `num_queries` (`int64`) |
| 28 | 4 bytes | `dim` (`int32`) |
| 32 | `nv × dim × 4` | Embeddings (`float32`, row-major) |
| + | `nv × 8` | IDs (`int64`) |
| + | `nq × dim × 4` | Queries (`float32`, row-major) |

---

## Configuration Reference

| Key | Type | Description |
|-----|------|-------------|
| `dataset_path` | string | Path to `.vsd` file produced by `prepare_dataset` |
| `output_dir` | string | Directory for the benchmark report |
| `num_shards` | int | Number of worker shards (and threads) |
| `top_k` | int | Nearest neighbours to retrieve per query |
| `batch_size` | int | Queries sent to workers per round-trip |