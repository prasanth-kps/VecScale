#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace vecscale {

using Vector = std::vector<float>;
using Matrix = std::vector<Vector>;
using IdArray = std::vector<std::int64_t>;

struct SearchResult {
    Matrix scores;
    std::vector<std::vector<std::size_t>> local_indices;
};

struct GlobalSearchResult {
    Matrix scores;
    std::vector<std::vector<std::int64_t>> ids;
};

struct Dataset {
    Matrix embeddings;
    Matrix queries;
    IdArray ids;
};

struct BenchmarkSummary {
    double throughput_qps{};
    double p50_ms{};
    double p95_ms{};
    std::size_t query_count{};
};

}  // namespace vecscale
