#include "vecscale/compute.hpp"
#include <algorithm>
#include <numeric>
#ifdef VECSCALE_USE_OPENMP
#  include <omp.h>
#endif

namespace vecscale {

TopKResult topk_cosine_similarity(
    const Matrix&               embeddings,
    const Matrix&               queries,
    int                         k,
    const std::vector<int64_t>& ids,
    bool                        use_omp
) {
    const int N = static_cast<int>(embeddings.rows());
    const int Q = static_cast<int>(queries.rows());
    k = std::min(k, N);

    // L2-normalise every row so dot product == cosine similarity
    Matrix norm_emb = embeddings.rowwise().normalized();
    Matrix norm_q   = queries.rowwise().normalized();

    // scores[q, i] = cosine similarity between query q and corpus vector i
    // Shape: (Q, N)
    Matrix scores = norm_q * norm_emb.transpose();

    TopKResult result;
    result.num_queries = Q;
    result.k           = k;
    result.indices.resize(static_cast<size_t>(Q) * k);
    result.scores .resize(static_cast<size_t>(Q) * k);

    // Per-query partial sort to extract top-k; O(N log k) per query.
    auto process_query = [&](int q) {
        std::vector<int> order(N);
        std::iota(order.begin(), order.end(), 0);
        const float* row = &scores(q, 0);  // RowMajor: row q is contiguous
        std::partial_sort(
            order.begin(), order.begin() + k, order.end(),
            [row](int a, int b) { return row[a] > row[b]; }
        );
        const size_t base = static_cast<size_t>(q) * k;
        for (int j = 0; j < k; ++j) {
            result.indices[base + j] = ids[order[j]];
            result.scores [base + j] = row[order[j]];
        }
    };

#ifdef VECSCALE_USE_OPENMP
    if (use_omp) {
        #pragma omp parallel for schedule(dynamic, 16)
        for (int q = 0; q < Q; ++q) process_query(q);
    } else {
        for (int q = 0; q < Q; ++q) process_query(q);
    }
#else
    (void)use_omp;
    for (int q = 0; q < Q; ++q) process_query(q);
#endif

    return result;
}

} // namespace vecscale
