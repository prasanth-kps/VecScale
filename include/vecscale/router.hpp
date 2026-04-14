#pragma once
#include "common.hpp"
#include <vector>

namespace vecscale {

/// Partitions a corpus into contiguous shards for distribution across workers.
///
/// Rows are divided as evenly as possible; any remainder is spread one-per-shard
/// across the first shards.  Shard data is returned as copies so each Worker
/// owns its slice independently.
class QueryRouter {
public:
    /// @param embeddings  full corpus matrix (not modified, must outlive the router)
    /// @param ids         global corpus IDs aligned with embedding rows
    /// @param num_shards  number of partitions to create
    QueryRouter(const Matrix&               embeddings,
                const std::vector<int64_t>& ids,
                int                         num_shards);

    int num_shards() const { return num_shards_; }

    /// Return a copy of the embedding rows assigned to shard @p shard_id.
    Matrix get_shard(int shard_id) const;

    /// Return the corpus IDs assigned to shard @p shard_id.
    std::vector<int64_t> get_shard_ids(int shard_id) const;

    /// Number of rows in shard @p shard_id.
    int shard_size(int shard_id) const { return sizes_[shard_id]; }

private:
    const Matrix&               embeddings_;
    const std::vector<int64_t>& ids_;
    int                         num_shards_;
    std::vector<int>            offsets_;   ///< start row index per shard
    std::vector<int>            sizes_;     ///< row count per shard
};

} // namespace vecscale
