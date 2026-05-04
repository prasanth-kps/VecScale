# VecScale

Distributed GPU ANN Retrieval Prototype for Real-Time Vector Search — **C++17 reference implementation** in this repository.

VecScale partitions a corpus into shards, runs approximate cosine top-k on each shard, merges into a global ranking, and compares against a single-node exact baseline (recall@k and throughput).

---

## Requirements

| Tool | Minimum version |
|------|----------------|
| CMake | 3.16 |
| GCC or Clang | C++17 (`std::filesystem`) |

```bash
# macOS
brew install cmake

# Ubuntu / Debian
sudo apt install cmake build-essential
```

---

## Project structure

```
VecScale/
├── CMakeLists.txt
├── configs/
│   └── default.json          # optional metadata (Slurm / docs); not read by C++ binaries
├── include/vecscale/         # legacy / design sketches — **not** used by CMake (headers live under src/vecscale/)
├── src/vecscale/
│   ├── *.hpp, *.cpp          # library: data I/O, compute, workers, router, benchmark, baselines
├── scripts/
│   ├── prepare_dataset.cpp   # CLI: write CSV dataset directory
│   └── run_demo.cpp          # CLI: run baseline + distributed + recall, write report
├── slurm/                    # example cluster jobs (adjust modules and paths)
└── tools/                    # older stubs — build targets use scripts/
```

---

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j"$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
```

Artifacts: `build/prepare_dataset`, `build/run_demo`.

---

## Dataset layout (CSV)

`prepare_dataset` writes a directory containing:

| File | Contents |
|------|----------|
| `embeddings.csv` | Corpus vectors, one row per vector, comma-separated floats |
| `queries.csv` | Query vectors, same layout |
| `ids.csv` | One global `int64` id per corpus row |

---

## Usage

### 1. Generate a dataset

```bash
./build/prepare_dataset <output_dir> [num_vectors] [num_queries] [dim] [seed]
```

Example:

```bash
./build/prepare_dataset data/dataset_cpp 100000 1000 128 42
```

### 2. Run the benchmark

```bash
./build/run_demo <dataset_dir> [num_shards] [top_k] [output_dir]
```

Defaults: `num_shards=4`, `top_k=10`, `output_dir=results`.

Example:

```bash
./build/run_demo data/dataset_cpp 4 10 results
```

Stages printed to stdout:

1. Single-node exact top-k (timed once for baseline QPS).
2. Distributed search with per-query latency sampling (same merge path as production).
3. Recall@k versus the exact baseline (should be `1.0` for this exact merge of per-shard exact top-k).

A text report is written to `<output_dir>/benchmark_report.txt`.

---

## Architecture

```
QueryRouter
  │  splits corpus into N contiguous shards
  │
  ├── Worker 0  ──┐
  ├── Worker 1  ──┤  each runs topk_cosine_similarity on its shard
  └── Worker N  ──┘
                  │
             merge_topk()
                  │
           global top-k result
                  │
         compare vs. exact baseline → recall@k
```

---

## Slurm

Scripts under `slurm/` assume the repo is checked out at `$HOME/VecScale` and load **site-specific** modules (example: `gcc/14.3.0`, `cmake/4.1.2`). If a job fails with *unknown module*, run `module spider gcc` (or ask your admin) and edit the `module load` lines to match your cluster. Older log files may show failures from obsolete module names (for example `gcc12.2.0` without the `gcc/` prefix).

---

## Python / other files

`src/vecscale/*.py` and `scripts/*_full.py` are experimental scaffolds and are **not** wired into the CMake build.
