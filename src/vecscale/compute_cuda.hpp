#pragma once

#include <string>

#include "vecscale/types.hpp"

namespace vecscale {

bool topk_cosine_similarity_cuda(
    const Matrix& queries,
    const Matrix& vectors,
    std::size_t top_k,
    SearchResult* out,
    std::string* error_message);

}  // namespace vecscale
