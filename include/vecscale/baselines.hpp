#pragma once
#include "common.hpp"
#include <string>

namespace vecscale {

/// Timing and throughput summary from the single-node baseline.
struct BaselineResult {
    TopKResult results;          ///< exact top-k for every query
    double     elapsed_ms = 0.0; ///< total wall-clock time
    double     qps        = 0.0; ///< queries per second
};

/// Brute-force single-node cosine similarity search with OpenMP parallelism.
///
/// This is the ground-truth reference: distributed results are compared against
/// this to compute recall@k.  Every result is exact (no approximation).
BaselineResult measure_single_node_baseline(
    const Matrix&               embeddings,
    const Matrix&               queries,
    const std::vector<int64_t>& ids,
    int                         k
);

} // namespace vecscale
