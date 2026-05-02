#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <mpi.h>

#include "vecscale/baselines.hpp"
#include "vecscale/benchmark.hpp"
#include "vecscale/data.hpp"
#include "vecscale/worker.hpp"

namespace {

vecscale::ComputeBackend parse_backend(const std::string& backend_text) {
    if (backend_text == "cpu") {
        return vecscale::ComputeBackend::Cpu;
    }
    if (backend_text == "openmp") {
        return vecscale::ComputeBackend::OpenMP;
    }
    if (backend_text == "cuda") {
        return vecscale::ComputeBackend::Cuda;
    }
    throw std::invalid_argument("Unknown backend. Supported values: cpu, openmp, cuda");
}

vecscale::GlobalSearchResult merge_rank_candidates(
    const std::vector<float>& gathered_scores,
    const std::vector<std::int64_t>& gathered_ids,
    int world_size,
    std::size_t query_count,
    std::size_t top_k) {
    vecscale::GlobalSearchResult out{};
    out.scores.resize(query_count);
    out.ids.resize(query_count);

    for (std::size_t q = 0; q < query_count; ++q) {
        std::vector<std::tuple<float, std::int64_t>> candidates;
        candidates.reserve(static_cast<std::size_t>(world_size) * top_k);
        for (int rank = 0; rank < world_size; ++rank) {
            const std::size_t base = (static_cast<std::size_t>(rank) * query_count + q) * top_k;
            for (std::size_t k = 0; k < top_k; ++k) {
                candidates.emplace_back(gathered_scores[base + k], gathered_ids[base + k]);
            }
        }
        std::partial_sort(
            candidates.begin(),
            candidates.begin() + static_cast<std::ptrdiff_t>(top_k),
            candidates.end(),
            [](const auto& lhs, const auto& rhs) { return std::get<0>(lhs) > std::get<0>(rhs); });
        for (std::size_t k = 0; k < top_k; ++k) {
            out.scores[q].push_back(std::get<0>(candidates[k]));
            out.ids[q].push_back(std::get<1>(candidates[k]));
        }
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc < 2) {
        if (rank == 0) {
            std::cerr << "Usage: run_demo_mpi <dataset_dir> [top_k] [backend] [omp_threads]\n";
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    try {
        const std::string dataset_dir = argv[1];
        const std::size_t top_k = (argc > 2) ? static_cast<std::size_t>(std::stoull(argv[2])) : 10;
        const std::string backend_text = (argc > 3) ? argv[3] : "cpu";
        const int omp_threads = (argc > 4) ? std::stoi(argv[4]) : 0;

        vecscale::RuntimeConfig config{};
        config.backend = parse_backend(backend_text);
        config.omp_threads = omp_threads;
        config.shard_parallel = false;

        const auto dataset = vecscale::load_dataset_csv(dataset_dir);
        if (dataset.embeddings.empty() || dataset.queries.empty()) {
            if (rank == 0) {
                std::cerr << "Dataset is empty. Generate data first.\n";
            }
            MPI_Finalize();
            return EXIT_FAILURE;
        }

        const std::size_t total_vectors = dataset.embeddings.size();
        const std::size_t query_count = dataset.queries.size();
        const std::size_t shard_size = (total_vectors + static_cast<std::size_t>(world_size) - 1) / static_cast<std::size_t>(world_size);
        const std::size_t begin = static_cast<std::size_t>(rank) * shard_size;
        const std::size_t end = std::min(begin + shard_size, total_vectors);

        vecscale::Matrix shard_embeddings;
        vecscale::IdArray shard_ids;
        if (begin < end) {
            shard_embeddings = vecscale::Matrix(
                dataset.embeddings.begin() + static_cast<std::ptrdiff_t>(begin),
                dataset.embeddings.begin() + static_cast<std::ptrdiff_t>(end));
            shard_ids = vecscale::IdArray(
                dataset.ids.begin() + static_cast<std::ptrdiff_t>(begin),
                dataset.ids.begin() + static_cast<std::ptrdiff_t>(end));
        } else {
            shard_embeddings = vecscale::Matrix(1, vecscale::Vector(dataset.embeddings[0].size(), 0.0f));
            shard_ids = vecscale::IdArray(1, -1);
        }

        vecscale::Worker local_worker(static_cast<std::size_t>(rank), std::move(shard_embeddings), std::move(shard_ids));

        const auto start = MPI_Wtime();
        const auto local = local_worker.search(dataset.queries, top_k, config);
        const auto stop = MPI_Wtime();
        const double local_compute_s = stop - start;

        std::vector<float> local_scores_flat(query_count * top_k, -1e9f);
        std::vector<std::int64_t> local_ids_flat(query_count * top_k, -1);
        for (std::size_t q = 0; q < query_count; ++q) {
            for (std::size_t k = 0; k < top_k; ++k) {
                if (k < local.scores[q].size()) {
                    local_scores_flat[q * top_k + k] = local.scores[q][k];
                }
                if (k < local.ids[q].size()) {
                    local_ids_flat[q * top_k + k] = local.ids[q][k];
                }
            }
        }

        std::vector<float> gathered_scores;
        std::vector<std::int64_t> gathered_ids;
        if (rank == 0) {
            gathered_scores.resize(static_cast<std::size_t>(world_size) * query_count * top_k, -1e9f);
            gathered_ids.resize(static_cast<std::size_t>(world_size) * query_count * top_k, -1);
        }

        MPI_Gather(
            local_scores_flat.data(),
            static_cast<int>(local_scores_flat.size()),
            MPI_FLOAT,
            gathered_scores.data(),
            static_cast<int>(query_count * top_k),
            MPI_FLOAT,
            0,
            MPI_COMM_WORLD);
        MPI_Gather(
            local_ids_flat.data(),
            static_cast<int>(local_ids_flat.size()),
            MPI_LONG_LONG_INT,
            gathered_ids.data(),
            static_cast<int>(query_count * top_k),
            MPI_LONG_LONG_INT,
            0,
            MPI_COMM_WORLD);

        double max_compute_s = 0.0;
        MPI_Reduce(&local_compute_s, &max_compute_s, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            const auto merged = merge_rank_candidates(gathered_scores, gathered_ids, world_size, query_count, top_k);
            (void)merged;
            const double distributed_qps = max_compute_s > 0.0 ? static_cast<double>(query_count) / max_compute_s : 0.0;
            const double baseline_qps =
                vecscale::measure_single_node_baseline_qps(dataset.queries, dataset.embeddings, dataset.ids, top_k);

            std::cout << "=== MPI Benchmark Summary ===\n";
            std::cout << "MPI ranks: " << world_size << "\n";
            std::cout << "Backend: " << backend_text << "\n";
            std::cout << "Queries: " << query_count << "\n";
            std::cout << "Baseline throughput (single-node exact): " << baseline_qps << " qps\n";
            std::cout << "Distributed throughput (MPI gather phase): " << distributed_qps << " qps\n";
            std::cout << "Top-K merge complete on rank 0.\n";
        }
    } catch (const std::exception& ex) {
        if (rank == 0) {
            std::cerr << "Error: " << ex.what() << "\n";
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
