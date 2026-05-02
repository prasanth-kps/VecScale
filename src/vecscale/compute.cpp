#include "vecscale/compute.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace vecscale {

namespace {

float dot(const Vector& a, const Vector& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector dimension mismatch in dot product.");
    }
    float sum = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

float norm(const Vector& v) {
    return std::sqrt(dot(v, v));
}

}  // namespace

SearchResult topk_cosine_similarity(const Matrix& queries, const Matrix& vectors, std::size_t top_k) {
    SearchResult out{};
    out.scores.resize(queries.size());
    out.local_indices.resize(queries.size());

    if (vectors.empty()) {
        return out;
    }

    const std::size_t k = std::min(top_k, vectors.size());
    std::vector<float> vector_norms(vectors.size(), 0.0f);
    for (std::size_t j = 0; j < vectors.size(); ++j) {
        vector_norms[j] = std::max(norm(vectors[j]), 1e-8f);
    }

    for (std::size_t qi = 0; qi < queries.size(); ++qi) {
        const float q_norm = std::max(norm(queries[qi]), 1e-8f);
        std::vector<std::pair<float, std::size_t>> scored;
        scored.reserve(vectors.size());

        for (std::size_t vj = 0; vj < vectors.size(); ++vj) {
            const float score = dot(queries[qi], vectors[vj]) / (q_norm * vector_norms[vj]);
            scored.emplace_back(score, vj);
        }

        std::partial_sort(
            scored.begin(),
            scored.begin() + static_cast<std::ptrdiff_t>(k),
            scored.end(),
            [](const auto& lhs, const auto& rhs) { return lhs.first > rhs.first; });

        out.scores[qi].reserve(k);
        out.local_indices[qi].reserve(k);
        for (std::size_t i = 0; i < k; ++i) {
            out.scores[qi].push_back(scored[i].first);
            out.local_indices[qi].push_back(scored[i].second);
        }
    }

    return out;
}

}  // namespace vecscale
