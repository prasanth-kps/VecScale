#pragma once

#include "vecscale/router.hpp"
#include "vecscale/types.hpp"

namespace vecscale {

BenchmarkSummary run_benchmark(QueryRouter& router, const Matrix& queries, std::size_t top_k);

}  // namespace vecscale
