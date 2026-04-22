#pragma once
#include "vecscale/types.hpp"
namespace vecscale {
GlobalSearchResult merge_topk(const std::vector<std::vector<std::vector<std::int64_t>>>&, const std::vector<Matrix>&, std::size_t);
}
