#pragma once

#include "vecscale/types.hpp"

namespace vecscale {

class Worker {
public:
    Worker(std::size_t worker_id, Matrix shard_embeddings, IdArray shard_ids);

    GlobalSearchResult search(const Matrix& queries, std::size_t top_k) const;
    std::size_t id() const { return worker_id_; }

private:
    std::size_t worker_id_{};
    Matrix embeddings_;
    IdArray ids_;
};

}  // namespace vecscale
