#pragma once

#include "vecscale/types.hpp"

namespace vecscale {

SearchResult topk_cosine_similarity(const Matrix& queries, const Matrix& vectors, std::size_t top_k);

}  // namespace vecscale
