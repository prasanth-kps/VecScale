#pragma once
#include <string>

namespace vecscale {

/// Runtime configuration loaded from a JSON file.
struct BenchmarkConfig {
    std::string dataset_path = "data/dataset.vsd";
    std::string output_dir   = "results";
    int         num_shards   = 4;
    int         top_k        = 10;
    int         batch_size   = 8;
};

/// Parse a JSON config file into a BenchmarkConfig.
BenchmarkConfig load_config(const std::string& path);

/// Aggregated benchmark output.
struct BenchmarkResult {
    double recall_at_k     = 0.0; ///< fraction of baseline top-k found by distributed search
    double distributed_qps = 0.0; ///< queries per second with sharded workers
    double baseline_qps    = 0.0; ///< queries per second with single-node brute force
    double p50_latency_ms  = 0.0; ///< median per-batch latency
    double p99_latency_ms  = 0.0; ///< 99th percentile per-batch latency
};

/// Run the full VecScale benchmark pipeline:
///   load dataset → baseline → distributed search → recall & latency report.
BenchmarkResult run_benchmark(const BenchmarkConfig& config);

} // namespace vecscale
