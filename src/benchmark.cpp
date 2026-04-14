#include "vecscale/benchmark.hpp"
#include "vecscale/data.hpp"
#include "vecscale/aggregator.hpp"
#include "vecscale/router.hpp"
#include "vecscale/worker.hpp"
#include "vecscale/baselines.hpp"

#include <nlohmann/json.hpp>
#include <fstream>
#include <thread>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <set>
#include <stdexcept>

namespace vecscale {

// ---- Config ----------------------------------------------------------------

BenchmarkConfig load_config(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open config: " + path);

    nlohmann::json j;
    f >> j;

    BenchmarkConfig cfg;
    if (j.contains("dataset_path")) cfg.dataset_path = j["dataset_path"].get<std::string>();
    if (j.contains("output_dir"))   cfg.output_dir   = j["output_dir"].get<std::string>();
    if (j.contains("num_shards"))   cfg.num_shards   = j["num_shards"].get<int>();
    if (j.contains("top_k"))        cfg.top_k        = j["top_k"].get<int>();
    if (j.contains("batch_size"))   cfg.batch_size   = j["batch_size"].get<int>();
    return cfg;
}

// ---- Recall ----------------------------------------------------------------

/// Compute mean recall@k: fraction of exact top-k found in approximate results.
static double compute_recall(const TopKResult& approx,
                              const TopKResult& exact,
                              int k) {
    const int Q = approx.num_queries;
    double total = 0.0;
    for (int q = 0; q < Q; ++q) {
        std::set<int64_t> exact_set(
            exact.indices.begin() + q * k,
            exact.indices.begin() + (q + 1) * k
        );
        int hits = 0;
        for (int j = 0; j < k; ++j)
            if (exact_set.count(approx.indices[q * k + j])) ++hits;
        total += static_cast<double>(hits) / k;
    }
    return total / Q;
}

// ---- Percentile helper -----------------------------------------------------

static double percentile(std::vector<double>& sorted_vals, double p) {
    if (sorted_vals.empty()) return 0.0;
    const size_t idx = static_cast<size_t>(std::ceil(p * sorted_vals.size())) - 1;
    return sorted_vals[std::min(idx, sorted_vals.size() - 1)];
}

// ---- Benchmark -------------------------------------------------------------

BenchmarkResult run_benchmark(const BenchmarkConfig& config) {
    std::cout << "=== VecScale Distributed Benchmark ===\n";

    // Load dataset
    std::cout << "Loading dataset: " << config.dataset_path << "\n";
    Dataset ds = load_dataset(config.dataset_path);

    const int N   = static_cast<int>(ds.embeddings.rows());
    const int Q   = static_cast<int>(ds.queries.rows());
    const int dim = static_cast<int>(ds.embeddings.cols());
    const int k   = config.top_k;

    std::cout << "  corpus=" << N
              << "  queries=" << Q
              << "  dim=" << dim
              << "  shards=" << config.num_shards
              << "  k=" << k
              << "  batch_size=" << config.batch_size << "\n\n";

    // ---- Single-node baseline ----
    std::cout << "[1/3] Single-node baseline (OpenMP)...\n";
    BaselineResult baseline =
        measure_single_node_baseline(ds.embeddings, ds.queries, ds.ids, k);
    std::cout << "  elapsed=" << baseline.elapsed_ms << " ms"
              << "  QPS=" << static_cast<int>(baseline.qps) << "\n\n";

    // ---- Build distributed workers ----
    std::cout << "[2/3] Distributed search (" << config.num_shards << " shards)...\n";
    QueryRouter router(ds.embeddings, ds.ids, config.num_shards);

    std::vector<Worker> workers;
    workers.reserve(config.num_shards);
    for (int i = 0; i < config.num_shards; ++i) {
        std::cout << "  shard " << i
                  << ": " << router.shard_size(i) << " vectors\n";
        workers.emplace_back(i, router.get_shard(i), router.get_shard_ids(i));
    }
    std::cout << "\n";

    // ---- Query batches ----
    std::vector<double> batch_latencies_ms;
    TopKResult distributed_result;
    distributed_result.num_queries = Q;
    distributed_result.k           = k;
    distributed_result.indices.resize(static_cast<size_t>(Q) * k);
    distributed_result.scores .resize(static_cast<size_t>(Q) * k);

    double total_ms = 0.0;

    for (int batch_start = 0; batch_start < Q; batch_start += config.batch_size) {
        const int batch_end = std::min(batch_start + config.batch_size, Q);
        const int bq        = batch_end - batch_start;
        Matrix batch_q      = ds.queries.middleRows(batch_start, bq);

        auto t0 = std::chrono::high_resolution_clock::now();

        // Each worker runs in its own thread
        std::vector<ShardResult> shard_results(config.num_shards);
        std::vector<std::thread> threads;
        threads.reserve(config.num_shards);
        for (int i = 0; i < config.num_shards; ++i)
            threads.emplace_back([&, i]() {
                shard_results[i] = workers[i].process_queries(batch_q, k);
            });
        for (auto& t : threads) t.join();

        TopKResult merged = merge_topk(shard_results, k);

        auto t1 = std::chrono::high_resolution_clock::now();
        const double batch_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        batch_latencies_ms.push_back(batch_ms);
        total_ms += batch_ms;

        // Stitch batch results into the full result array
        const size_t src_end = static_cast<size_t>(bq) * k;
        std::copy(merged.indices.begin(), merged.indices.begin() + src_end,
                  distributed_result.indices.begin() + batch_start * k);
        std::copy(merged.scores .begin(), merged.scores .begin() + src_end,
                  distributed_result.scores .begin() + batch_start * k);
    }

    const double dist_qps = (static_cast<double>(Q) / total_ms) * 1000.0;
    std::sort(batch_latencies_ms.begin(), batch_latencies_ms.end());
    const double p50 = percentile(batch_latencies_ms, 0.50);
    const double p99 = percentile(batch_latencies_ms, 0.99);

    std::cout << "  elapsed=" << total_ms << " ms"
              << "  QPS=" << static_cast<int>(dist_qps) << "\n"
              << "  p50=" << p50 << " ms"
              << "  p99=" << p99 << " ms\n\n";

    // ---- Recall ----
    std::cout << "[3/3] Computing recall@" << k << "...\n";
    const double recall = compute_recall(distributed_result, baseline.results, k);
    std::cout << "  recall@" << k << " = " << recall << "\n\n";

    // ---- Write report ----
    std::filesystem::create_directories(config.output_dir);
    const std::string report_path = config.output_dir + "/benchmark_report.txt";
    {
        std::ofstream rep(report_path);
        rep << "VecScale Benchmark Report\n"
            << "=========================\n"
            << "corpus_size     : " << N            << "\n"
            << "num_queries     : " << Q            << "\n"
            << "dim             : " << dim           << "\n"
            << "num_shards      : " << config.num_shards << "\n"
            << "top_k           : " << k            << "\n"
            << "baseline_qps    : " << baseline.qps << "\n"
            << "distributed_qps : " << dist_qps     << "\n"
            << "p50_latency_ms  : " << p50           << "\n"
            << "p99_latency_ms  : " << p99           << "\n"
            << "recall_at_k     : " << recall        << "\n";
    }
    std::cout << "Report written to: " << report_path << "\n";

    return {recall, dist_qps, baseline.qps, p50, p99};
}

} // namespace vecscale
