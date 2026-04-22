#include "vecscale/aggregator.hpp"
#include <algorithm>
#include <stdexcept>
#include <tuple>
namespace vecscale {
GlobalSearchResult merge_topk(const std::vector<std::vector<std::vector<std::int64_t>>>& shard_ids, const std::vector<Matrix>& shard_scores, std::size_t top_k){
    if (shard_ids.size()!=shard_scores.size()) throw std::invalid_argument("mismatch");
    GlobalSearchResult out; if (shard_scores.empty()) return out; std::size_t qn = shard_scores.front().size(); out.ids.resize(qn); out.scores.resize(qn);
    for(std::size_t q=0;q<qn;++q){ std::vector<std::tuple<float,std::int64_t>> all; for(std::size_t s=0;s<shard_scores.size();++s){ if(q>=shard_scores[s].size()||q>=shard_ids[s].size()) continue; auto& sc=shard_scores[s][q]; auto& id=shard_ids[s][q]; std::size_t c=std::min(sc.size(), id.size()); for(std::size_t i=0;i<c;++i) all.emplace_back(sc[i], id[i]); }
        std::size_t k=std::min(top_k, all.size()); std::partial_sort(all.begin(), all.begin()+static_cast<std::ptrdiff_t>(k), all.end(), [](auto&a,auto&b){return std::get<0>(a)>std::get<0>(b);});
        for(std::size_t i=0;i<k;++i){ out.scores[q].push_back(std::get<0>(all[i])); out.ids[q].push_back(std::get<1>(all[i])); }} return out; }
}
