#pragma once

#include <string>

#include "vecscale/types.hpp"

namespace vecscale {

Dataset generate_synthetic_dataset(std::size_t num_vectors, std::size_t num_queries, std::size_t dim, unsigned seed);
void save_dataset_csv(const Dataset& dataset, const std::string& output_dir);
Dataset load_dataset_csv(const std::string& input_dir);

}  // namespace vecscale
