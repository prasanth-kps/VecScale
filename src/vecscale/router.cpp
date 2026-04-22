#include "vecscale/router.hpp"
#include "vecscale/aggregator.hpp"
namespace vecscale {
QueryRouter::QueryRouter(std::vector<Worker> workers): workers_(std::move(workers)) {}
GlobalSearchResult QueryRouter::search(const Matrix& queries, std::size_t top_k) const {
    std::vector<std::vector<std::vector<std::int64_t>>> shard_ids; std::vector<Matrix> shard_scores;
    for (const auto& w : workers_) { auto r = w.search(queries, top_k); shard_ids.push_back(r.ids); shard_scores.push_back(r.scores); }
    return merge_topk(shard_ids, shard_scores, top_k);
}
}
