#include "vecscale/baselines.hpp"
#include <chrono>
#include "vecscale/compute.hpp"
namespace vecscale {
GlobalSearchResult exact_baseline_topk(const Matrix& q, const Matrix& e, const IdArray& ids, std::size_t k){ auto local=topk_cosine_similarity(q,e,k); GlobalSearchResult out; out.scores=local.scores; out.ids.resize(local.local_indices.size()); for(std::size_t i=0;i<local.local_indices.size();++i) for(auto idx:local.local_indices[i]) out.ids[i].push_back(ids[idx]); return out; }
double measure_single_node_baseline_qps(const Matrix& q, const Matrix& e, const IdArray& ids, std::size_t k){ auto s=std::chrono::high_resolution_clock::now(); (void)exact_baseline_topk(q,e,ids,k); auto t=std::chrono::high_resolution_clock::now(); std::chrono::duration<double> d=t-s; return d.count()>0.0? static_cast<double>(q.size())/d.count() : 0.0; }
}
