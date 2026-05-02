#include "vecscale/benchmark.hpp"

#include <algorithm>
#include <chrono>
#include <vector>

namespace vecscale {

BenchmarkSummary run_benchmark(QueryRouter& router, const Matrix& queries, std::size_t top_k) {
    std::vector<double> latencies_ms;
    latencies_ms.reserve(queries.size());

    const auto start_total = std::chrono::high_resolution_clock::now();
    for (const auto& query : queries) {
        const Matrix single_query{query};
        const auto start = std::chrono::high_resolution_clock::now();
        (void)router.search(single_query, top_k);
        const auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> elapsed_ms = end - start;
        latencies_ms.push_back(elapsed_ms.count());
    }
    const auto end_total = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> total_s = end_total - start_total;

    std::sort(latencies_ms.begin(), latencies_ms.end());
    const auto p50_idx = static_cast<std::size_t>(0.50 * (latencies_ms.size() - 1));
    const auto p95_idx = static_cast<std::size_t>(0.95 * (latencies_ms.size() - 1));

    BenchmarkSummary summary{};
    summary.query_count = queries.size();
    summary.throughput_qps = total_s.count() > 0.0 ? static_cast<double>(queries.size()) / total_s.count() : 0.0;
    summary.p50_ms = latencies_ms.empty() ? 0.0 : latencies_ms[p50_idx];
    summary.p95_ms = latencies_ms.empty() ? 0.0 : latencies_ms[p95_idx];
    return summary;
}

}  // namespace vecscale
