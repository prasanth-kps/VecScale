#pragma once
#include "vecscale/types.hpp"
#include "vecscale/worker.hpp"
namespace vecscale {
class QueryRouter {
public:
    explicit QueryRouter(std::vector<Worker> workers);
    GlobalSearchResult search(const Matrix&, std::size_t) const;
private:
    std::vector<Worker> workers_;
};
}
