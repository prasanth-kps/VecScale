#pragma once
#include "vecscale/router.hpp"
#include "vecscale/types.hpp"
namespace vecscale {
BenchmarkSummary run_benchmark(QueryRouter&, const Matrix&, std::size_t);
}
