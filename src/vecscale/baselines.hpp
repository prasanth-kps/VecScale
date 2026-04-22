#pragma once
#include "vecscale/types.hpp"
namespace vecscale {
GlobalSearchResult exact_baseline_topk(const Matrix&, const Matrix&, const IdArray&, std::size_t);
double measure_single_node_baseline_qps(const Matrix&, const Matrix&, const IdArray&, std::size_t);
}
