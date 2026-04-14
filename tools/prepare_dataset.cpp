/// prepare_dataset — generate a synthetic embedding dataset and save it
/// as a VecScale binary (.vsd) file for use with run_demo.
///
/// Usage:
///   prepare_dataset --output <path.vsd>
///                  [--num-vectors N]   (default: 100000)
///                  [--num-queries N]   (default: 1000)
///                  [--dim D]           (default: 128)
///                  [--seed S]          (default: 42)

#include "vecscale/data.hpp"
#include <iostream>
#include <string>
#include <random>
#include <cstdint>
#include <stdexcept>

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " --output <path>"
              << " [--num-vectors N] [--num-queries N] [--dim D] [--seed S]\n";
}

int main(int argc, char* argv[]) {
    std::string output;
    int64_t num_vectors = 100'000;
    int64_t num_queries = 1'000;
    int     dim         = 128;
    uint64_t seed       = 42;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "--output"      && i+1 < argc) output       = argv[++i];
        else if (arg == "--num-vectors" && i+1 < argc) num_vectors  = std::stoll(argv[++i]);
        else if (arg == "--num-queries" && i+1 < argc) num_queries  = std::stoll(argv[++i]);
        else if (arg == "--dim"         && i+1 < argc) dim          = std::stoi (argv[++i]);
        else if (arg == "--seed"        && i+1 < argc) seed         = std::stoull(argv[++i]);
        else { print_usage(argv[0]); return 1; }
    }
    if (output.empty()) { print_usage(argv[0]); return 1; }

    try {
        std::mt19937_64 rng(seed);
        std::normal_distribution<float> dist(0.0f, 1.0f);

        vecscale::Dataset ds;
        ds.embeddings.resize(num_vectors, dim);
        ds.queries.resize(num_queries, dim);
        ds.ids.resize(num_vectors);

        std::cout << "Generating " << num_vectors
                  << " corpus vectors (dim=" << dim << ")...\n";
        for (int64_t i = 0; i < num_vectors; ++i) {
            ds.ids[i] = i;
            for (int d = 0; d < dim; ++d)
                ds.embeddings(i, d) = dist(rng);
        }

        std::cout << "Generating " << num_queries << " query vectors...\n";
        for (int64_t i = 0; i < num_queries; ++i)
            for (int d = 0; d < dim; ++d)
                ds.queries(i, d) = dist(rng);

        vecscale::save_dataset(output, ds);

        std::cout << "Dataset saved: " << output << "\n"
                  << "  embeddings : (" << num_vectors << ", " << dim << ")\n"
                  << "  queries    : (" << num_queries << ", " << dim << ")\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
