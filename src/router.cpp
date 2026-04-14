#include "vecscale/router.hpp"
#include <stdexcept>

namespace vecscale {

QueryRouter::QueryRouter(
    const Matrix&               embeddings,
    const std::vector<int64_t>& ids,
    int                         num_shards
)
    : embeddings_(embeddings), ids_(ids), num_shards_(num_shards)
{
    if (num_shards <= 0)
        throw std::invalid_argument("num_shards must be positive");
    if (static_cast<int>(ids.size()) != embeddings.rows())
        throw std::invalid_argument("ids.size() must equal embeddings.rows()");

    const int N          = static_cast<int>(embeddings.rows());
    const int base_size  = N / num_shards;
    const int remainder  = N % num_shards;

    offsets_.resize(num_shards);
    sizes_  .resize(num_shards);

    int offset = 0;
    for (int i = 0; i < num_shards; ++i) {
        offsets_[i] = offset;
        sizes_  [i] = base_size + (i < remainder ? 1 : 0);
        offset      += sizes_[i];
    }
}

Matrix QueryRouter::get_shard(int shard_id) const {
    return embeddings_.middleRows(offsets_[shard_id], sizes_[shard_id]);
}

std::vector<int64_t> QueryRouter::get_shard_ids(int shard_id) const {
    const int off = offsets_[shard_id];
    return std::vector<int64_t>(ids_.begin() + off,
                                ids_.begin() + off + sizes_[shard_id]);
}

} // namespace vecscale
