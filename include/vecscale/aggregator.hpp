#pragma once
#include "common.hpp"
#include <vector>

namespace vecscale {

/// Merge per-shard top-k candidate lists into a single global top-k result.
///
/// Each ShardResult holds the best k candidates from one shard for every query.
/// The function pools all candidates and selects the globally best k across shards.
///
/// @param shard_results  one ShardResult per worker shard
/// @param k              desired number of final results per query
/// @returns TopKResult with globally ranked top-k for every query
TopKResult merge_topk(const std::vector<ShardResult>& shard_results, int k);

} // namespace vecscale
