#pragma once
#include <Eigen/Dense>
#include <vector>
#include <cstdint>

namespace vecscale {

/// Row-major float matrix — each row is one vector in the corpus or query set.
using Matrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

/// Top-K result for a batch of Q queries.
/// Layout: indices[q*k + j] is the j-th nearest-neighbour ID for query q.
struct TopKResult {
    std::vector<int64_t> indices;   ///< global corpus IDs  (size = num_queries * k)
    std::vector<float>   scores;    ///< cosine similarities (size = num_queries * k)
    int num_queries = 0;
    int k           = 0;
};

/// Candidate results returned by a single shard worker.
struct ShardResult {
    std::vector<int64_t> ids;       ///< shard-local top-k IDs per query (size = num_queries * k)
    std::vector<float>   scores;    ///< corresponding cosine similarities
    int num_queries = 0;
    int k           = 0;
};

} // namespace vecscale
