#include "vecscale/worker.hpp"
#include "vecscale/compute.hpp"
#include <iostream>

namespace vecscale {

Worker::Worker(int id, Matrix shard, std::vector<int64_t> shard_ids)
    : id_(id), shard_(std::move(shard)), shard_ids_(std::move(shard_ids))
{}

ShardResult Worker::process_queries(const Matrix& queries, int k) const {
    // OpenMP is disabled here: the benchmark launches one std::thread per worker,
    // so outer thread parallelism already provides concurrency.
    TopKResult r = topk_cosine_similarity(shard_, queries, k, shard_ids_,
                                          /*use_omp=*/false);
    ShardResult sr;
    sr.ids         = std::move(r.indices);
    sr.scores      = std::move(r.scores);
    sr.num_queries = r.num_queries;
    sr.k           = r.k;
    return sr;
}

void run_worker_server(int worker_id, Matrix shard, std::vector<int64_t> ids) {
    const int shard_size = static_cast<int>(shard.rows());
    Worker w(worker_id, std::move(shard), std::move(ids));
    std::cout << "[Worker " << worker_id << "] ready"
              << "  shard_size=" << shard_size << "\n";
    // In a real deployment: start gRPC/ZMQ server loop here.
    // In the in-process prototype, workers are driven directly by run_benchmark().
}

} // namespace vecscale
