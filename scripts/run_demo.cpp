#include <cstdlib>
#include <iostream>
#include <vector>
#include "vecscale/baselines.hpp"
#include "vecscale/benchmark.hpp"
#include "vecscale/data.hpp"
#include "vecscale/router.hpp"
#include "vecscale/worker.hpp"
int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "Usage: run_demo <dataset_dir> [num_shards] [top_k]\n"; return EXIT_FAILURE; }
    std::string dataset_dir = argv[1];
    std::size_t shards = (argc > 2) ? static_cast<std::size_t>(std::stoull(argv[2])) : 4;
    std::size_t top_k = (argc > 3) ? static_cast<std::size_t>(std::stoull(argv[3])) : 10;
    auto ds = vecscale::load_dataset_csv(dataset_dir);
    if (ds.embeddings.empty() || ds.queries.empty()) { std::cerr << "Dataset is empty.\n"; return EXIT_FAILURE; }
    std::vector<vecscale::Worker> workers;
    std::size_t shard_size = (ds.embeddings.size() + shards - 1) / shards;
    for (std::size_t s = 0; s < shards; ++s) {
        std::size_t b = s * shard_size, e = std::min(b + shard_size, ds.embeddings.size());
        if (b >= e) break;
        vecscale::Matrix em(ds.embeddings.begin() + static_cast<std::ptrdiff_t>(b), ds.embeddings.begin() + static_cast<std::ptrdiff_t>(e));
        vecscale::IdArray id(ds.ids.begin() + static_cast<std::ptrdiff_t>(b), ds.ids.begin() + static_cast<std::ptrdiff_t>(e));
        workers.emplace_back(s, std::move(em), std::move(id));
    }
    vecscale::QueryRouter router(std::move(workers));
    auto sum = vecscale::run_benchmark(router, ds.queries, top_k);
    double base = vecscale::measure_single_node_baseline_qps(ds.queries, ds.embeddings, ds.ids, top_k);
    std::cout << "queries=" << sum.query_count << " baseline_qps=" << base << " dist_qps=" << sum.throughput_qps << " p50=" << sum.p50_ms << " p95=" << sum.p95_ms << "\n";
    return EXIT_SUCCESS;
}
