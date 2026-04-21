#pragma once

#include "vecscale/types.hpp"
#include "vecscale/worker.hpp"

namespace vecscale {

class QueryRouter {
public:
    explicit QueryRouter(std::vector<Worker> workers);

    GlobalSearchResult search(const Matrix& queries, std::size_t top_k) const;

private:
    std::vector<Worker> workers_;
};

}  // namespace vecscale
