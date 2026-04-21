#include "vecscale/aggregator.hpp"

#include <algorithm>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace vecscale {

GlobalSearchResult merge_topk(
    const std::vector<std::vector<std::vector<std::int64_t>>>& shard_ids,
    const std::vector<Matrix>& shard_scores,
    std::size_t top_k) {
    if (shard_ids.size() != shard_scores.size()) {
        throw std::invalid_argument("shard_ids and shard_scores size mismatch.");
    }

    GlobalSearchResult out{};
    if (shard_scores.empty()) {
        return out;
    }

    const std::size_t query_count = shard_scores.front().size();
    out.ids.resize(query_count);
    out.scores.resize(query_count);

    for (std::size_t q = 0; q < query_count; ++q) {
        std::vector<std::tuple<float, std::int64_t>> merged;

        for (std::size_t s = 0; s < shard_scores.size(); ++s) {
            if (q >= shard_scores[s].size()) {
                continue;
            }
            const auto& scores_q = shard_scores[s][q];
            const auto& ids_s = shard_ids[s][q];
            const std::size_t count = std::min(scores_q.size(), ids_s.size());
            for (std::size_t i = 0; i < count; ++i) {
                merged.emplace_back(scores_q[i], ids_s[i]);
            }
        }

        const std::size_t k = std::min(top_k, merged.size());
        std::partial_sort(
            merged.begin(),
            merged.begin() + static_cast<std::ptrdiff_t>(k),
            merged.end(),
            [](const auto& lhs, const auto& rhs) { return std::get<0>(lhs) > std::get<0>(rhs); });

        out.ids[q].reserve(k);
        out.scores[q].reserve(k);
        for (std::size_t i = 0; i < k; ++i) {
            out.scores[q].push_back(std::get<0>(merged[i]));
            out.ids[q].push_back(std::get<1>(merged[i]));
        }
    }
    return out;
}

}  // namespace vecscale
