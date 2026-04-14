#pragma once
#include "common.hpp"

namespace vecscale {

/// A single shard worker.
///
/// Owns its slice of the corpus and answers query batches synchronously via
/// process_queries().  In the benchmark, each Worker runs inside its own
/// std::thread; inter-thread parallelism replaces OpenMP within the shard.
class Worker {
public:
    Worker(int id, Matrix shard, std::vector<int64_t> shard_ids);

    /// Run top-k cosine similarity on this shard for the given query batch.
    /// OpenMP is intentionally disabled here; the caller's thread pool provides
    /// the outer concurrency.
    ShardResult process_queries(const Matrix& queries, int k) const;

    int id()         const { return id_; }
    int shard_rows() const { return static_cast<int>(shard_.rows()); }

private:
    int                  id_;
    Matrix               shard_;
    std::vector<int64_t> shard_ids_;
};

/// Standalone entry-point matching the Python scaffold.
/// Constructs a Worker and prints a ready message.  In a production deployment
/// this would start a network server loop (e.g. gRPC).
void run_worker_server(int worker_id, Matrix shard, std::vector<int64_t> ids);

} // namespace vecscale
