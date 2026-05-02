#include <cstdlib>
#include <iostream>
#include <string>

#include "vecscale/data.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: prepare_dataset <output_dir> [num_vectors] [num_queries] [dim] [seed]\n";
        return EXIT_FAILURE;
    }

    const std::string output_dir = argv[1];
    const std::size_t num_vectors = (argc > 2) ? static_cast<std::size_t>(std::stoull(argv[2])) : 100000;
    const std::size_t num_queries = (argc > 3) ? static_cast<std::size_t>(std::stoull(argv[3])) : 1000;
    const std::size_t dim = (argc > 4) ? static_cast<std::size_t>(std::stoull(argv[4])) : 128;
    const unsigned seed = (argc > 5) ? static_cast<unsigned>(std::stoul(argv[5])) : 42U;

    const auto dataset = vecscale::generate_synthetic_dataset(num_vectors, num_queries, dim, seed);
    vecscale::save_dataset_csv(dataset, output_dir);

    std::cout << "Dataset written to " << output_dir << "\n";
    std::cout << "embeddings=" << dataset.embeddings.size() << " x " << dim << "\n";
    std::cout << "queries=" << dataset.queries.size() << " x " << dim << "\n";
    return EXIT_SUCCESS;
}
