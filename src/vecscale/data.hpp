#pragma once
#include <string>
#include "vecscale/types.hpp"
namespace vecscale {
Dataset generate_synthetic_dataset(std::size_t, std::size_t, std::size_t, unsigned);
void save_dataset_csv(const Dataset&, const std::string&);
Dataset load_dataset_csv(const std::string&);
}
