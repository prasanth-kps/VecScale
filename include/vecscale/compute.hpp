#pragma once
#include "common.hpp"

namespace vecscale {

/// Compute cosine similarity between every query and every embedding row,
/// and return the top-k nearest neighbours for each query.
///
/// @param embeddings  (num_vectors, dim) corpus matrix
/// @param queries     (num_queries, dim) query matrix
/// @param k           number of nearest neighbours to return per query
/// @param ids         global corpus IDs, one per embedding row
/// @param use_omp     enable OpenMP parallelism over the query loop (default: true)
/// @returns TopKResult with indices and scores arrays of shape (num_queries * k)
TopKResult topk_cosine_similarity(
    const Matrix&               embeddings,
    const Matrix&               queries,
    int                         k,
    const std::vector<int64_t>& ids,
    bool                        use_omp = true
);

} // namespace vecscale
