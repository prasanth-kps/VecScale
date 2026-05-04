#include "vecscale/router.hpp"

#include <mutex>
#include <thread>
#include <vector>

#include "vecscale/aggregator.hpp"

namespace vecscale {

QueryRouter::QueryRouter(std::vector<Worker> workers) : workers_(std::move(workers)) {}

GlobalSearchResult QueryRouter::search(const Matrix& queries, std::size_t top_k) const {
    const std::size_t n = workers_.size();

    std::vector<std::vector<std::vector<std::int64_t>>> shard_ids(n);
    std::vector<Matrix> shard_scores(n);

    // Each worker runs in its own thread; results land in pre-sized slots (no mutex needed).
    std::vector<std::thread> threads;
    threads.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        threads.emplace_back([&, i]() {
            const auto local = workers_[i].search(queries, top_k);
            shard_ids[i]     = local.ids;
            shard_scores[i]  = local.scores;
        });
    }
    for (auto& t : threads) {
        t.join();
    }

    return merge_topk(shard_ids, shard_scores, top_k);
}

}  // namespace vecscale
