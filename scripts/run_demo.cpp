#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "vecscale/baselines.hpp"
#include "vecscale/benchmark.hpp"
#include "vecscale/data.hpp"
#include "vecscale/router.hpp"
#include "vecscale/worker.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: run_demo <dataset_dir> [num_shards] [top_k] [output_dir]\n";
        return EXIT_FAILURE;
    }

    const std::string dataset_dir = argv[1];
    const std::size_t num_shards = (argc > 2) ? static_cast<std::size_t>(std::stoull(argv[2])) : 4;
    const std::size_t top_k = (argc > 3) ? static_cast<std::size_t>(std::stoull(argv[3])) : 10;
    const std::string output_dir = (argc > 4) ? argv[4] : "results";

    const auto dataset = vecscale::load_dataset_csv(dataset_dir);
    if (dataset.embeddings.empty() || dataset.queries.empty()) {
        std::cerr << "Dataset is empty. Generate data first.\n";
        return EXIT_FAILURE;
    }

    std::vector<vecscale::Worker> workers;
    const std::size_t shard_size = (dataset.embeddings.size() + num_shards - 1) / num_shards;
    workers.reserve(num_shards);
    for (std::size_t shard = 0; shard < num_shards; ++shard) {
        const std::size_t begin = shard * shard_size;
        const std::size_t end = std::min(begin + shard_size, dataset.embeddings.size());
        if (begin >= end) {
            break;
        }
        vecscale::Matrix shard_embeddings(dataset.embeddings.begin() + static_cast<std::ptrdiff_t>(begin),
                                          dataset.embeddings.begin() + static_cast<std::ptrdiff_t>(end));
        vecscale::IdArray shard_ids(dataset.ids.begin() + static_cast<std::ptrdiff_t>(begin),
                                    dataset.ids.begin() + static_cast<std::ptrdiff_t>(end));
        workers.emplace_back(shard, std::move(shard_embeddings), std::move(shard_ids));
    }

    vecscale::QueryRouter router(std::move(workers));

    const auto baseline_start = std::chrono::high_resolution_clock::now();
    const auto baseline = vecscale::exact_baseline_topk(
        dataset.queries, dataset.embeddings, dataset.ids, top_k);
    const auto baseline_end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> baseline_elapsed = baseline_end - baseline_start;
    const double baseline_qps =
        baseline_elapsed.count() > 0.0
            ? static_cast<double>(dataset.queries.size()) / baseline_elapsed.count()
            : 0.0;

    const auto dist_start = std::chrono::high_resolution_clock::now();
    const auto distributed = router.search(dataset.queries, top_k);
    const auto dist_end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> dist_elapsed = dist_end - dist_start;
    const double dist_qps =
        dist_elapsed.count() > 0.0
            ? static_cast<double>(dataset.queries.size()) / dist_elapsed.count()
            : 0.0;
    const double recall = vecscale::mean_recall_at_k(distributed, baseline, top_k);

    const auto summary = vecscale::run_benchmark(router, dataset.queries, top_k);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "=== VecScale Benchmark ===\n";
    std::cout << "corpus=" << dataset.embeddings.size() << "  queries=" << dataset.queries.size()
              << "  dim=" << (dataset.embeddings.empty() ? 0 : dataset.embeddings[0].size()) << "  shards="
              << router.worker_count() << "  k=" << top_k << "\n\n";
    std::cout << "[1/3] Single-node baseline (exact top-k)...\n";
    std::cout << "  elapsed=" << baseline_elapsed.count() * 1000.0 << " ms"
              << "  throughput=" << baseline_qps << " qps\n\n";
    std::cout << "[2/3] Distributed search (" << router.worker_count() << " parallel workers)...\n";
    std::cout << "  scatter+gather=" << dist_elapsed.count() * 1000.0 << " ms"
              << "  throughput=" << dist_qps << " qps\n";
    std::cout << "  per-query p50=" << summary.p50_ms << " ms  p95=" << summary.p95_ms << " ms"
              << "  throughput=" << summary.throughput_qps << " qps\n\n";
    std::cout << "[3/3] Recall vs. exact baseline...\n";
    std::cout << "  recall@" << top_k << " = " << std::setprecision(4) << recall << "\n";

    std::filesystem::create_directories(output_dir);
    const std::string report_path = output_dir + "/benchmark_report.txt";
    std::ofstream report(report_path);
    if (report.is_open()) {
        report << std::fixed << std::setprecision(6);
        report << "VecScale Benchmark Report\n";
        report << "=========================\n";
        report << "corpus_size     : " << dataset.embeddings.size() << "\n";
        report << "num_queries     : " << dataset.queries.size() << "\n";
        report << "dim             : " << (dataset.embeddings.empty() ? 0 : dataset.embeddings[0].size()) << "\n";
        report << "num_shards      : " << router.worker_count() << "\n";
        report << "top_k           : " << top_k << "\n";
        report << "baseline_qps    : " << baseline_qps << "\n";
        report << "distributed_qps : " << summary.throughput_qps << "\n";
        report << "dist_bulk_qps   : " << dist_qps << "\n";
        report << "p50_latency_ms  : " << summary.p50_ms << "\n";
        report << "p95_latency_ms  : " << summary.p95_ms << "\n";
        report << "recall_at_k     : " << recall << "\n";
        report.close();
        std::cout << "\nReport written to: " << report_path << "\n";
    } else {
        std::cerr << "Warning: could not write " << report_path << "\n";
    }
    return EXIT_SUCCESS;
}
