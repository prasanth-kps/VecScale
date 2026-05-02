#include <cstdlib>
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
        std::cerr << "Usage: run_demo <dataset_dir> [num_shards] [top_k]\n";
        return EXIT_FAILURE;
    }

    const std::string dataset_dir = argv[1];
    const std::size_t num_shards = (argc > 2) ? static_cast<std::size_t>(std::stoull(argv[2])) : 4;
    const std::size_t top_k = (argc > 3) ? static_cast<std::size_t>(std::stoull(argv[3])) : 10;

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
    const auto summary = vecscale::run_benchmark(router, dataset.queries, top_k);
    const double baseline_qps = vecscale::measure_single_node_baseline_qps(
        dataset.queries, dataset.embeddings, dataset.ids, top_k);

    std::cout << "=== C++ Benchmark Summary ===\n";
    std::cout << "Queries: " << summary.query_count << "\n";
    std::cout << "Baseline throughput (single-node exact): " << baseline_qps << " qps\n";
    std::cout << "Distributed throughput: " << summary.throughput_qps << " qps\n";
    std::cout << "p50 latency: " << summary.p50_ms << " ms\n";
    std::cout << "p95 latency: " << summary.p95_ms << " ms\n";
    return EXIT_SUCCESS;
}
