#include "vecscale/worker.hpp"
namespace vecscale {
Worker::Worker(std::size_t worker_id, Matrix shard_embeddings, IdArray shard_ids): worker_id_(worker_id), embeddings_(std::move(shard_embeddings)), ids_(std::move(shard_ids)) {}
GlobalSearchResult Worker::search(const Matrix& queries, std::size_t top_k) const { GlobalSearchResult out; out.ids.assign(queries.size(), std::vector<std::int64_t>(top_k, ids_.empty() ? -1 : ids_[0])); out.scores.assign(queries.size(), std::vector<float>(top_k, 0.0f)); return out; }
}
