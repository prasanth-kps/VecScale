#include "vecscale/benchmark.hpp"

#include <algorithm>
#include <chrono>
#include <vector>

#ifdef VECSCALE_HAS_OPENMP
#include <omp.h>
#endif

namespace vecscale {

BenchmarkSummary run_benchmark(
    QueryRouter& router,
    const Matrix& queries,
    std::size_t top_k,
    const RuntimeConfig& config) {
    std::vector<double> latencies_ms(queries.size(), 0.0);

    const auto start_total = std::chrono::high_resolution_clock::now();
    bool parallel_queries = false;
#ifdef VECSCALE_HAS_OPENMP
    parallel_queries = config.backend == ComputeBackend::OpenMP;
    if (parallel_queries && config.omp_threads > 0) {
        omp_set_num_threads(config.omp_threads);
    }
#pragma omp parallel for if(parallel_queries)
#endif
    for (std::ptrdiff_t qi = 0; qi < static_cast<std::ptrdiff_t>(queries.size()); ++qi) {
        const Matrix single_query{queries[static_cast<std::size_t>(qi)]};
        const auto start = std::chrono::high_resolution_clock::now();
        (void)router.search(single_query, top_k, config);
        const auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> elapsed_ms = end - start;
        latencies_ms[static_cast<std::size_t>(qi)] = elapsed_ms.count();
    }
    const auto end_total = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> total_s = end_total - start_total;

    if (!latencies_ms.empty()) {
        std::sort(latencies_ms.begin(), latencies_ms.end());
    }
    const auto p50_idx = latencies_ms.empty() ? 0 : static_cast<std::size_t>(0.50 * (latencies_ms.size() - 1));
    const auto p95_idx = latencies_ms.empty() ? 0 : static_cast<std::size_t>(0.95 * (latencies_ms.size() - 1));

    BenchmarkSummary summary{};
    summary.query_count = queries.size();
    summary.throughput_qps = total_s.count() > 0.0 ? static_cast<double>(queries.size()) / total_s.count() : 0.0;
    summary.p50_ms = latencies_ms.empty() ? 0.0 : latencies_ms[p50_idx];
    summary.p95_ms = latencies_ms.empty() ? 0.0 : latencies_ms[p95_idx];
    return summary;
}

}  // namespace vecscale
