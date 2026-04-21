#pragma once

#include "vecscale/types.hpp"

namespace vecscale {

GlobalSearchResult merge_topk(
    const std::vector<std::vector<std::vector<std::int64_t>>>& shard_ids,
    const std::vector<Matrix>& shard_scores,
    std::size_t top_k);

}  // namespace vecscale
