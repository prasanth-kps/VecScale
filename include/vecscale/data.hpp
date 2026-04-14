#pragma once
#include "common.hpp"
#include <string>

namespace vecscale {

/// In-memory representation of a labelled embedding dataset.
struct Dataset {
    Matrix               embeddings;   ///< (num_vectors, dim) corpus embeddings
    Matrix               queries;      ///< (num_queries,  dim) query embeddings
    std::vector<int64_t> ids;          ///< global IDs for each corpus vector
};

/// Persist a Dataset to a binary .vsd file (VecScale Data format).
/// Creates parent directories automatically.
void save_dataset(const std::string& path, const Dataset& ds);

/// Load a Dataset previously written by save_dataset().
Dataset load_dataset(const std::string& path);

} // namespace vecscale
