#include "vecscale/baselines.hpp"

#include <chrono>

#include "vecscale/compute.hpp"

namespace vecscale {

GlobalSearchResult exact_baseline_topk(
    const Matrix& queries,
    const Matrix& embeddings,
    const IdArray& ids,
    std::size_t top_k) {
    const SearchResult local = topk_cosine_similarity(queries, embeddings, top_k);
    GlobalSearchResult out{};
    out.scores = local.scores;
    out.ids.resize(local.local_indices.size());

    for (std::size_t q = 0; q < local.local_indices.size(); ++q) {
        for (const auto idx : local.local_indices[q]) {
            out.ids[q].push_back(ids[idx]);
        }
    }
    return out;
}

double measure_single_node_baseline_qps(
    const Matrix& queries,
    const Matrix& embeddings,
    const IdArray& ids,
    std::size_t top_k) {
    const auto start = std::chrono::high_resolution_clock::now();
    (void)exact_baseline_topk(queries, embeddings, ids, top_k);
    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> elapsed = end - start;
    return elapsed.count() > 0.0 ? static_cast<double>(queries.size()) / elapsed.count() : 0.0;
}

}  // namespace vecscale
