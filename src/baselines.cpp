#include "vecscale/baselines.hpp"
#include "vecscale/compute.hpp"
#include <chrono>

namespace vecscale {

BaselineResult measure_single_node_baseline(
    const Matrix&               embeddings,
    const Matrix&               queries,
    const std::vector<int64_t>& ids,
    int                         k
) {
    auto t0 = std::chrono::high_resolution_clock::now();

    TopKResult results = topk_cosine_similarity(
        embeddings, queries, k, ids, /*use_omp=*/true
    );

    auto t1 = std::chrono::high_resolution_clock::now();

    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double qps =
        (static_cast<double>(queries.rows()) / elapsed_ms) * 1000.0;

    return {std::move(results), elapsed_ms, qps};
}

} // namespace vecscale
