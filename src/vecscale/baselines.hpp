#pragma once

#include "vecscale/types.hpp"

namespace vecscale {

GlobalSearchResult exact_baseline_topk(const Matrix& queries, const Matrix& embeddings, const IdArray& ids, std::size_t top_k);
double measure_single_node_baseline_qps(const Matrix& queries, const Matrix& embeddings, const IdArray& ids, std::size_t top_k);

}  // namespace vecscale
