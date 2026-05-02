#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <string>
#include <stdexcept>
#include <vector>

#include "vecscale/baselines.hpp"
#include "vecscale/benchmark.hpp"
#include "vecscale/data.hpp"
#include "vecscale/router.hpp"
#include "vecscale/worker.hpp"

namespace {

vecscale::ComputeBackend parse_backend(const std::string& backend_text) {
    if (backend_text == "cpu") {
        return vecscale::ComputeBackend::Cpu;
    }
    if (backend_text == "openmp") {
        return vecscale::ComputeBackend::OpenMP;
    }
    if (backend_text == "cuda") {
        return vecscale::ComputeBackend::Cuda;
    }
    throw std::invalid_argument("Unknown backend. Supported values: cpu, openmp, cuda");
}

std::vector<std::size_t> parse_shards_mode(std::string_view mode_text, std::size_t default_shards) {
    if (mode_text == "single") {
        return {default_shards};
    }
    if (mode_text == "scale") {
        return {1, 2, 4};
    }
    throw std::invalid_argument("Unknown mode. Supported values: single, scale");
}

vecscale::QueryRouter build_router(const vecscale::Dataset& dataset, std::size_t num_shards) {
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
    return vecscale::QueryRouter(std::move(workers));
}

const char* backend_name(vecscale::ComputeBackend backend) {
    switch (backend) {
    case vecscale::ComputeBackend::Cpu:
        return "cpu";
    case vecscale::ComputeBackend::OpenMP:
        return "openmp";
    case vecscale::ComputeBackend::Cuda:
        return "cuda";
    }
    return "cpu";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: run_demo <dataset_dir> [num_shards] [top_k] [backend] [omp_threads] [mode]\n";
        std::cerr << "  backend: cpu | openmp | cuda\n";
        std::cerr << "  mode: single | scale\n";
        return EXIT_FAILURE;
    }
    const std::string dataset_dir = argv[1];
    const std::size_t num_shards = (argc > 2) ? static_cast<std::size_t>(std::stoull(argv[2])) : 4;
    const std::size_t top_k = (argc > 3) ? static_cast<std::size_t>(std::stoull(argv[3])) : 10;
    const std::string backend_text = (argc > 4) ? argv[4] : "cpu";
    const int omp_threads = (argc > 5) ? std::stoi(argv[5]) : 0;
    const std::string mode_text = (argc > 6) ? argv[6] : "single";

    const auto dataset = vecscale::load_dataset_csv(dataset_dir);
    if (dataset.embeddings.empty() || dataset.queries.empty()) {
        std::cerr << "Dataset is empty. Generate data first.\n";
        return EXIT_FAILURE;
    }

    vecscale::RuntimeConfig config{};
    config.backend = parse_backend(backend_text);
    config.omp_threads = omp_threads;
    config.shard_parallel = config.backend == vecscale::ComputeBackend::OpenMP;

    const auto baseline_qps = vecscale::measure_single_node_baseline_qps(
        dataset.queries, dataset.embeddings, dataset.ids, top_k);
    const auto shard_sweep = parse_shards_mode(mode_text, num_shards);

    std::cout << "=== C++ Benchmark Summary ===\n";
    std::cout << "Backend: " << backend_name(config.backend) << "\n";
    std::cout << "OMP threads: " << config.omp_threads << "\n";
    std::cout << "Queries: " << dataset.queries.size() << "\n";
    std::cout << "Baseline throughput (single-node exact): " << baseline_qps << " qps\n";

    for (std::size_t shards : shard_sweep) {
        auto router = build_router(dataset, shards);
        const auto summary = vecscale::run_benchmark(router, dataset.queries, top_k, config);
        std::cout << "shards=" << shards
                  << " distributed_qps=" << summary.throughput_qps
                  << " p50_ms=" << summary.p50_ms
                  << " p95_ms=" << summary.p95_ms << "\n";
    }
    return EXIT_SUCCESS;
}
