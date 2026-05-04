#pragma once

#include "vecscale/types.hpp"

namespace vecscale {

GlobalSearchResult exact_baseline_topk(const Matrix& queries, const Matrix& embeddings, const IdArray& ids, std::size_t top_k);
double measure_single_node_baseline_qps(const Matrix& queries, const Matrix& embeddings, const IdArray& ids, std::size_t top_k);

/// Mean per-query recall: fraction of exact top-k neighbour IDs that appear in the approximate top-k lists.
double mean_recall_at_k(
    const GlobalSearchResult& approx,
    const GlobalSearchResult& exact,
    std::size_t k);

}  // namespace vecscale
