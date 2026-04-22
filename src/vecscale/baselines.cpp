#include "vecscale/baselines.hpp"
namespace vecscale {
GlobalSearchResult exact_baseline_topk(const Matrix&, const Matrix&, const IdArray&, std::size_t) { return {}; }
double measure_single_node_baseline_qps(const Matrix&, const Matrix&, const IdArray&, std::size_t) { return 0.0; }
}
