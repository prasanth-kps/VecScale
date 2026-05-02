#include "vecscale/compute_cuda.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

namespace vecscale {

namespace {

__global__ void cosine_scores_kernel(
    const float* queries,
    const float* vectors,
    const float* query_norms,
    const float* vector_norms,
    float* scores,
    int query_count,
    int vector_count,
    int dim) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = query_count * vector_count;
    if (idx >= total) {
        return;
    }

    const int q = idx / vector_count;
    const int v = idx % vector_count;

    float dot = 0.0f;
    const int q_offset = q * dim;
    const int v_offset = v * dim;
    for (int i = 0; i < dim; ++i) {
        dot += queries[q_offset + i] * vectors[v_offset + i];
    }

    const float denom = fmaxf(query_norms[q] * vector_norms[v], 1e-8f);
    scores[idx] = dot / denom;
}

float l2_norm(const Vector& vec) {
    float sum = 0.0f;
    for (float value : vec) {
        sum += value * value;
    }
    return std::sqrt(sum);
}

}  // namespace

bool topk_cosine_similarity_cuda(
    const Matrix& queries,
    const Matrix& vectors,
    std::size_t top_k,
    SearchResult* out,
    std::string* error_message) {
    if (out == nullptr) {
        if (error_message != nullptr) {
            *error_message = "Null output pointer for CUDA compute path.";
        }
        return false;
    }
    out->scores.clear();
    out->local_indices.clear();

    if (queries.empty() || vectors.empty()) {
        out->scores.resize(queries.size());
        out->local_indices.resize(queries.size());
        return true;
    }

    const int query_count = static_cast<int>(queries.size());
    const int vector_count = static_cast<int>(vectors.size());
    const int dim = static_cast<int>(vectors[0].size());
    const std::size_t k = std::min(top_k, vectors.size());

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count <= 0) {
        if (error_message != nullptr) {
            *error_message = "No CUDA devices visible at runtime.";
        }
        return false;
    }

    std::vector<float> h_queries(static_cast<std::size_t>(query_count) * dim);
    std::vector<float> h_vectors(static_cast<std::size_t>(vector_count) * dim);
    std::vector<float> h_q_norms(query_count, 0.0f);
    std::vector<float> h_v_norms(vector_count, 0.0f);

    for (int q = 0; q < query_count; ++q) {
        if (static_cast<int>(queries[q].size()) != dim) {
            if (error_message != nullptr) {
                *error_message = "Query dimension mismatch for CUDA path.";
            }
            return false;
        }
        for (int d = 0; d < dim; ++d) {
            h_queries[static_cast<std::size_t>(q) * dim + d] = queries[q][d];
        }
        h_q_norms[q] = std::max(l2_norm(queries[q]), 1e-8f);
    }
    for (int v = 0; v < vector_count; ++v) {
        if (static_cast<int>(vectors[v].size()) != dim) {
            if (error_message != nullptr) {
                *error_message = "Vector dimension mismatch for CUDA path.";
            }
            return false;
        }
        for (int d = 0; d < dim; ++d) {
            h_vectors[static_cast<std::size_t>(v) * dim + d] = vectors[v][d];
        }
        h_v_norms[v] = std::max(l2_norm(vectors[v]), 1e-8f);
    }

    float* d_queries = nullptr;
    float* d_vectors = nullptr;
    float* d_q_norms = nullptr;
    float* d_v_norms = nullptr;
    float* d_scores = nullptr;

    const std::size_t queries_bytes = h_queries.size() * sizeof(float);
    const std::size_t vectors_bytes = h_vectors.size() * sizeof(float);
    const std::size_t q_norm_bytes = h_q_norms.size() * sizeof(float);
    const std::size_t v_norm_bytes = h_v_norms.size() * sizeof(float);
    const std::size_t scores_bytes = static_cast<std::size_t>(query_count) * vector_count * sizeof(float);

    auto cleanup = [&]() {
        cudaFree(d_queries);
        cudaFree(d_vectors);
        cudaFree(d_q_norms);
        cudaFree(d_v_norms);
        cudaFree(d_scores);
    };

    if (cudaMalloc(&d_queries, queries_bytes) != cudaSuccess ||
        cudaMalloc(&d_vectors, vectors_bytes) != cudaSuccess ||
        cudaMalloc(&d_q_norms, q_norm_bytes) != cudaSuccess ||
        cudaMalloc(&d_v_norms, v_norm_bytes) != cudaSuccess ||
        cudaMalloc(&d_scores, scores_bytes) != cudaSuccess) {
        cleanup();
        if (error_message != nullptr) {
            *error_message = "CUDA allocation failed.";
        }
        return false;
    }

    if (cudaMemcpy(d_queries, h_queries.data(), queries_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_vectors, h_vectors.data(), vectors_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_q_norms, h_q_norms.data(), q_norm_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_v_norms, h_v_norms.data(), v_norm_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cleanup();
        if (error_message != nullptr) {
            *error_message = "CUDA memcpy host-to-device failed.";
        }
        return false;
    }

    const int total = query_count * vector_count;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    cosine_scores_kernel<<<blocks, threads>>>(
        d_queries, d_vectors, d_q_norms, d_v_norms, d_scores, query_count, vector_count, dim);

    if (cudaGetLastError() != cudaSuccess) {
        cleanup();
        if (error_message != nullptr) {
            *error_message = "CUDA kernel launch failed.";
        }
        return false;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        cleanup();
        if (error_message != nullptr) {
            *error_message = "CUDA kernel synchronize failed.";
        }
        return false;
    }

    std::vector<float> h_scores(static_cast<std::size_t>(query_count) * vector_count, 0.0f);
    if (cudaMemcpy(h_scores.data(), d_scores, scores_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        cleanup();
        if (error_message != nullptr) {
            *error_message = "CUDA memcpy device-to-host failed.";
        }
        return false;
    }

    cleanup();

    out->scores.resize(query_count);
    out->local_indices.resize(query_count);
    for (int q = 0; q < query_count; ++q) {
        std::vector<std::pair<float, std::size_t>> scored;
        scored.reserve(vector_count);
        for (int v = 0; v < vector_count; ++v) {
            scored.emplace_back(h_scores[static_cast<std::size_t>(q) * vector_count + v], static_cast<std::size_t>(v));
        }
        std::partial_sort(
            scored.begin(),
            scored.begin() + static_cast<std::ptrdiff_t>(k),
            scored.end(),
            [](const auto& lhs, const auto& rhs) { return lhs.first > rhs.first; });

        out->scores[q].reserve(k);
        out->local_indices[q].reserve(k);
        for (std::size_t i = 0; i < k; ++i) {
            out->scores[q].push_back(scored[i].first);
            out->local_indices[q].push_back(scored[i].second);
        }
    }
    return true;
}

}  // namespace vecscale
