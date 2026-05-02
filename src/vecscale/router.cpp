#include "vecscale/router.hpp"

#include "vecscale/aggregator.hpp"

#ifdef VECSCALE_HAS_OPENMP
#include <omp.h>
#endif

namespace vecscale {

QueryRouter::QueryRouter(std::vector<Worker> workers) : workers_(std::move(workers)) {}

GlobalSearchResult QueryRouter::search(const Matrix& queries, std::size_t top_k, const RuntimeConfig& config) const {
    std::vector<std::vector<std::vector<std::int64_t>>> shard_ids(workers_.size());
    std::vector<Matrix> shard_scores(workers_.size());

    bool parallel_shards = false;
#ifdef VECSCALE_HAS_OPENMP
    parallel_shards = config.shard_parallel;
    if (parallel_shards && config.omp_threads > 0) {
        omp_set_num_threads(config.omp_threads);
    }
#pragma omp parallel for if(parallel_shards)
#endif
    for (std::ptrdiff_t wi = 0; wi < static_cast<std::ptrdiff_t>(workers_.size()); ++wi) {
        const auto local = workers_[static_cast<std::size_t>(wi)].search(queries, top_k, config);
        shard_ids[static_cast<std::size_t>(wi)] = std::move(local.ids);
        shard_scores[static_cast<std::size_t>(wi)] = std::move(local.scores);
    }

    return merge_topk(shard_ids, shard_scores, top_k);
}

}  // namespace vecscale
