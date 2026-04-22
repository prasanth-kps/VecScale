#include "vecscale/worker.hpp"
#include <stdexcept>
#include "vecscale/compute.hpp"
namespace vecscale {
Worker::Worker(std::size_t worker_id, Matrix shard_embeddings, IdArray shard_ids): worker_id_(worker_id), embeddings_(std::move(shard_embeddings)), ids_(std::move(shard_ids)) { if (embeddings_.size()!=ids_.size()) throw std::invalid_argument("mismatch"); }
GlobalSearchResult Worker::search(const Matrix& queries, std::size_t top_k) const {
    auto local = topk_cosine_similarity(queries, embeddings_, top_k); GlobalSearchResult out; out.scores = local.scores; out.ids.resize(local.local_indices.size());
    for(std::size_t q=0;q<local.local_indices.size();++q) for(auto idx: local.local_indices[q]) out.ids[q].push_back(ids_[idx]);
    return out;
}
}
