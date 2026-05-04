#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "vecscale/baselines.hpp"
#include "vecscale/benchmark.hpp"
#include "vecscale/data.hpp"
#include "vecscale/router.hpp"
#include "vecscale/worker.hpp"

namespace {

vecscale::ComputeBackend parse_backend(const std::string& text) {
    if (text == "cpu")    { return vecscale::ComputeBackend::Cpu; }
    if (text == "openmp") { return vecscale::ComputeBackend::OpenMP; }
    if (text == "cuda")   { return vecscale::ComputeBackend::Cuda; }
    throw std::invalid_argument("Unknown backend. Supported: cpu | openmp | cuda");
}

const char* backend_name(vecscale::ComputeBackend b) {
    switch (b) {
    case vecscale::ComputeBackend::Cpu:    return "cpu";
    case vecscale::ComputeBackend::OpenMP: return "openmp";
    case vecscale::ComputeBackend::Cuda:   return "cuda";
    }
    return "cpu";
}

vecscale::QueryRouter build_router(const vecscale::Dataset& dataset, std::size_t num_shards) {
    std::vector<vecscale::Worker> workers;
    const std::size_t shard_size = (dataset.embeddings.size() + num_shards - 1) / num_shards;
    workers.reserve(num_shards);
    for (std::size_t s = 0; s < num_shards; ++s) {
        const std::size_t begin = s * shard_size;
        const std::size_t end   = std::min(begin + shard_size, dataset.embeddings.size());
        if (begin >= end) { break; }
        vecscale::Matrix shard_emb(
            dataset.embeddings.begin() + static_cast<std::ptrdiff_t>(begin),
            dataset.embeddings.begin() + static_cast<std::ptrdiff_t>(end));
        vecscale::IdArray shard_ids(
            dataset.ids.begin() + static_cast<std::ptrdiff_t>(begin),
            dataset.ids.begin() + static_cast<std::ptrdiff_t>(end));
        workers.emplace_back(s, std::move(shard_emb), std::move(shard_ids));
    }
    return vecscale::QueryRouter(std::move(workers));
}

void write_report(
    const std::string& output_dir,
    std::size_t corpus_size,
    std::size_t num_queries,
    std::size_t dim,
    std::size_t num_shards,
    std::size_t top_k,
    const char* backend,
    double baseline_qps,
    double dist_bulk_qps,
    double per_query_qps,
    double p50_ms,
    double p95_ms,
    double recall) {
    std::filesystem::create_directories(output_dir);
    const std::string path = output_dir + "/benchmark_report.txt";
    std::ofstream f(path);
    if (!f.is_open()) {
        std::cerr << "Warning: could not write " << path << "\n";
        return;
    }
    f << std::fixed << std::setprecision(6);
    f << "VecScale Benchmark Report\n";
    f << "=========================\n";
    f << "corpus_size     : " << corpus_size     << "\n";
    f << "num_queries     : " << num_queries      << "\n";
    f << "dim             : " << dim              << "\n";
    f << "num_shards      : " << num_shards       << "\n";
    f << "top_k           : " << top_k            << "\n";
    f << "backend         : " << backend          << "\n";
    f << "baseline_qps    : " << baseline_qps     << "\n";
    f << "dist_bulk_qps   : " << dist_bulk_qps    << "\n";
    f << "distributed_qps : " << per_query_qps    << "\n";
    f << "p50_latency_ms  : " << p50_ms           << "\n";
    f << "p95_latency_ms  : " << p95_ms           << "\n";
    f << "recall_at_k     : " << recall           << "\n";
    f.close();
    std::cout << "Report written to: " << path << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: run_demo <dataset_dir> [num_shards] [top_k] [backend] [omp_threads] [output_dir]\n";
        std::cerr << "  backend:    cpu | openmp | cuda   (default: cpu)\n";
        return EXIT_FAILURE;
    }

    const std::string dataset_dir  = argv[1];
    const std::size_t num_shards   = (argc > 2) ? static_cast<std::size_t>(std::stoull(argv[2])) : 4;
    const std::size_t top_k        = (argc > 3) ? static_cast<std::size_t>(std::stoull(argv[3])) : 10;
    const std::string backend_text = (argc > 4) ? argv[4] : "cpu";
    const int omp_threads          = (argc > 5) ? std::stoi(argv[5]) : 0;
    const std::string output_dir   = (argc > 6) ? argv[6] : "results";

    vecscale::RuntimeConfig config{};
    config.backend      = parse_backend(backend_text);
    config.omp_threads  = omp_threads;
    config.shard_parallel = false;  // std::thread handles shard parallelism in router

    const auto dataset = vecscale::load_dataset_csv(dataset_dir);
    if (dataset.embeddings.empty() || dataset.queries.empty()) {
        std::cerr << "Dataset is empty. Generate data first.\n";
        return EXIT_FAILURE;
    }

    const std::size_t dim = dataset.embeddings.empty() ? 0 : dataset.embeddings[0].size();

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "=== VecScale Benchmark ===\n";
    std::cout << "corpus=" << dataset.embeddings.size()
              << "  queries=" << dataset.queries.size()
              << "  dim=" << dim
              << "  shards=" << num_shards
              << "  k=" << top_k
              << "  backend=" << backend_name(config.backend) << "\n\n";

    // Stage 1: exact single-node baseline (timed once).
    std::cout << "[1/3] Single-node baseline (exact top-k)...\n";
    const auto t0 = std::chrono::high_resolution_clock::now();
    const auto baseline = vecscale::exact_baseline_topk(
        dataset.queries, dataset.embeddings, dataset.ids, top_k);
    const auto t1 = std::chrono::high_resolution_clock::now();
    const double baseline_elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double baseline_qps = baseline_elapsed_ms > 0.0
        ? static_cast<double>(dataset.queries.size()) / (baseline_elapsed_ms / 1000.0) : 0.0;
    std::cout << "  elapsed=" << baseline_elapsed_ms << " ms"
              << "  throughput=" << baseline_qps << " qps\n\n";

    // Stage 2: distributed search (all queries in one scatter-gather, then per-query latency).
    std::cout << "[2/3] Distributed search (" << num_shards << " parallel shards)...\n";
    auto router = build_router(dataset, num_shards);

    const auto t2 = std::chrono::high_resolution_clock::now();
    const auto distributed = router.search(dataset.queries, top_k, config);
    const auto t3 = std::chrono::high_resolution_clock::now();
    const double bulk_ms  = std::chrono::duration<double, std::milli>(t3 - t2).count();
    const double bulk_qps = bulk_ms > 0.0
        ? static_cast<double>(dataset.queries.size()) / (bulk_ms / 1000.0) : 0.0;

    const auto summary = vecscale::run_benchmark(router, dataset.queries, top_k, config);
    std::cout << "  scatter+gather=" << bulk_ms << " ms"
              << "  bulk_qps=" << bulk_qps << "\n";
    std::cout << "  per-query p50=" << summary.p50_ms << " ms"
              << "  p95=" << summary.p95_ms << " ms"
              << "  throughput=" << summary.throughput_qps << " qps\n\n";

    // Stage 3: recall vs exact baseline.
    std::cout << "[3/3] Recall vs. exact baseline...\n";
    const double recall = vecscale::mean_recall_at_k(distributed, baseline, top_k);
    std::cout << "  recall@" << top_k << " = " << std::setprecision(4) << recall << "\n\n";

    write_report(output_dir,
        dataset.embeddings.size(), dataset.queries.size(), dim, router.worker_count(), top_k,
        backend_name(config.backend),
        baseline_qps, bulk_qps, summary.throughput_qps,
        summary.p50_ms, summary.p95_ms, recall);

    return EXIT_SUCCESS;
}
