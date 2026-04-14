#include "vecscale/aggregator.hpp"
#include <algorithm>
#include <stdexcept>

namespace vecscale {

TopKResult merge_topk(const std::vector<ShardResult>& shard_results, int k) {
    if (shard_results.empty())
        throw std::runtime_error("merge_topk: empty shard_results");

    const int Q = shard_results[0].num_queries;
    for (const auto& sr : shard_results)
        if (sr.num_queries != Q)
            throw std::runtime_error("merge_topk: inconsistent num_queries across shards");

    TopKResult merged;
    merged.num_queries = Q;
    merged.k           = k;
    merged.indices.resize(static_cast<size_t>(Q) * k);
    merged.scores .resize(static_cast<size_t>(Q) * k);

    for (int q = 0; q < Q; ++q) {
        // Pool candidates from every shard for this query
        std::vector<std::pair<float, int64_t>> candidates;
        candidates.reserve(shard_results.size() * static_cast<size_t>(shard_results[0].k));

        for (const auto& sr : shard_results) {
            const size_t base = static_cast<size_t>(q) * sr.k;
            for (int j = 0; j < sr.k; ++j)
                candidates.emplace_back(sr.scores[base + j], sr.ids[base + j]);
        }

        // Select global top-k by descending score — O(C log k) where C = num_shards * shard_k
        const int take = std::min(k, static_cast<int>(candidates.size()));
        std::partial_sort(
            candidates.begin(), candidates.begin() + take, candidates.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; }
        );

        const size_t base = static_cast<size_t>(q) * k;
        for (int j = 0; j < take; ++j) {
            merged.scores [base + j] = candidates[j].first;
            merged.indices[base + j] = candidates[j].second;
        }
    }

    return merged;
}

} // namespace vecscale
