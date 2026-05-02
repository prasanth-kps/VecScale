#include "vecscale/compute.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>

#include "vecscale/compute_cuda.hpp"

#ifdef VECSCALE_HAS_OPENMP
#include <omp.h>
#endif

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

SearchResult topk_cosine_similarity_cpu(
    const Matrix& queries,
    const Matrix& vectors,
    std::size_t top_k,
    bool use_openmp,
    int omp_threads) {
    SearchResult out{};
    out.scores.resize(queries.size());
    out.local_indices.resize(queries.size());

    if (vectors.empty()) {
        return out;
    }

    const std::size_t k = std::min(top_k, vectors.size());
    std::vector<float> vector_norms(vectors.size(), 0.0f);

#ifdef VECSCALE_HAS_OPENMP
    if (use_openmp && omp_threads > 0) {
        omp_set_num_threads(omp_threads);
    }
#pragma omp parallel for if(use_openmp)
#endif
    for (std::ptrdiff_t j = 0; j < static_cast<std::ptrdiff_t>(vectors.size()); ++j) {
        vector_norms[static_cast<std::size_t>(j)] = std::max(norm(vectors[static_cast<std::size_t>(j)]), 1e-8f);
    }

#ifdef VECSCALE_HAS_OPENMP
#pragma omp parallel for if(use_openmp)
#endif
    for (std::ptrdiff_t qi = 0; qi < static_cast<std::ptrdiff_t>(queries.size()); ++qi) {
        const std::size_t q_idx = static_cast<std::size_t>(qi);
        const float q_norm = std::max(norm(queries[q_idx]), 1e-8f);
        std::vector<std::pair<float, std::size_t>> scored;
        scored.reserve(vectors.size());

        for (std::size_t vj = 0; vj < vectors.size(); ++vj) {
            const float score = dot(queries[q_idx], vectors[vj]) / (q_norm * vector_norms[vj]);
            scored.emplace_back(score, vj);
        }

        std::partial_sort(
            scored.begin(),
            scored.begin() + static_cast<std::ptrdiff_t>(k),
            scored.end(),
            [](const auto& lhs, const auto& rhs) { return lhs.first > rhs.first; });

        out.scores[q_idx].reserve(k);
        out.local_indices[q_idx].reserve(k);
        for (std::size_t i = 0; i < k; ++i) {
            out.scores[q_idx].push_back(scored[i].first);
            out.local_indices[q_idx].push_back(scored[i].second);
        }
    }

    return out;
}

}  // namespace

SearchResult topk_cosine_similarity(
    const Matrix& queries,
    const Matrix& vectors,
    std::size_t top_k,
    const RuntimeConfig& config) {
    if (config.backend == ComputeBackend::Cuda) {
        SearchResult out{};
        std::string cuda_error;
        if (topk_cosine_similarity_cuda(queries, vectors, top_k, &out, &cuda_error)) {
            return out;
        }
        static bool warned_once = false;
        if (!warned_once) {
            std::cerr << "[vecscale] CUDA backend unavailable, falling back to CPU/OpenMP: " << cuda_error << "\n";
            warned_once = true;
        }
    }

    bool use_openmp = false;
#ifdef VECSCALE_HAS_OPENMP
    use_openmp = config.backend == ComputeBackend::OpenMP;
#endif
    return topk_cosine_similarity_cpu(queries, vectors, top_k, use_openmp, config.omp_threads);
}

}  // namespace vecscale
