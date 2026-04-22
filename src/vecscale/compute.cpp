#include "vecscale/compute.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace vecscale {
namespace {
float dot(const Vector& a, const Vector& b) { if (a.size()!=b.size()) throw std::invalid_argument("dim"); float s=0.0f; for(std::size_t i=0;i<a.size();++i) s+=a[i]*b[i]; return s; }
float nrm(const Vector& v){ return std::sqrt(dot(v,v)); }
}
SearchResult topk_cosine_similarity(const Matrix& queries, const Matrix& vectors, std::size_t top_k){
    SearchResult out; out.scores.resize(queries.size()); out.local_indices.resize(queries.size());
    if (vectors.empty()) return out;
    std::size_t k = std::min(top_k, vectors.size());
    std::vector<float> vnorm(vectors.size(), 0.0f); for(std::size_t j=0;j<vectors.size();++j) vnorm[j]=std::max(nrm(vectors[j]),1e-8f);
    for(std::size_t qi=0; qi<queries.size(); ++qi){
        float qn = std::max(nrm(queries[qi]),1e-8f);
        std::vector<std::pair<float,std::size_t>> scored; scored.reserve(vectors.size());
        for(std::size_t vj=0; vj<vectors.size(); ++vj) scored.emplace_back(dot(queries[qi], vectors[vj])/(qn*vnorm[vj]), vj);
        std::partial_sort(scored.begin(), scored.begin()+static_cast<std::ptrdiff_t>(k), scored.end(), [](auto&a,auto&b){return a.first>b.first;});
        for(std::size_t i=0;i<k;++i){ out.scores[qi].push_back(scored[i].first); out.local_indices[qi].push_back(scored[i].second);} }
    return out;
}
}
