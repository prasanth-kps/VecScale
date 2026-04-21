#include "vecscale/data.hpp"

#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>

namespace vecscale {

namespace {

void write_matrix_csv(const Matrix& matrix, const std::string& path) {
    std::ofstream out(path);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open file for write: " + path);
    }
    for (const auto& row : matrix) {
        for (std::size_t i = 0; i < row.size(); ++i) {
            out << row[i];
            if (i + 1 < row.size()) {
                out << ",";
            }
        }
        out << "\n";
    }
}

Matrix read_matrix_csv(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open file for read: " + path);
    }
    Matrix out;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        std::stringstream ss(line);
        std::string token;
        Vector row;
        while (std::getline(ss, token, ',')) {
            row.push_back(std::stof(token));
        }
        out.push_back(std::move(row));
    }
    return out;
}

}  // namespace

Dataset generate_synthetic_dataset(std::size_t num_vectors, std::size_t num_queries, std::size_t dim, unsigned seed) {
    Dataset ds{};
    ds.embeddings.assign(num_vectors, Vector(dim, 0.0f));
    ds.queries.assign(num_queries, Vector(dim, 0.0f));
    ds.ids.resize(num_vectors);

    std::mt19937 rng(seed);
    std::normal_distribution<float> gaussian(0.0f, 1.0f);

    for (std::size_t i = 0; i < num_vectors; ++i) {
        ds.ids[i] = static_cast<std::int64_t>(i);
        for (std::size_t j = 0; j < dim; ++j) {
            ds.embeddings[i][j] = gaussian(rng);
        }
    }
    for (std::size_t i = 0; i < num_queries; ++i) {
        for (std::size_t j = 0; j < dim; ++j) {
            ds.queries[i][j] = gaussian(rng);
        }
    }
    return ds;
}

void save_dataset_csv(const Dataset& dataset, const std::string& output_dir) {
    std::filesystem::create_directories(output_dir);
    write_matrix_csv(dataset.embeddings, output_dir + "/embeddings.csv");
    write_matrix_csv(dataset.queries, output_dir + "/queries.csv");

    std::ofstream ids_out(output_dir + "/ids.csv");
    if (!ids_out.is_open()) {
        throw std::runtime_error("Failed to open ids.csv for write.");
    }
    for (const auto id : dataset.ids) {
        ids_out << id << "\n";
    }
}

Dataset load_dataset_csv(const std::string& input_dir) {
    Dataset ds{};
    ds.embeddings = read_matrix_csv(input_dir + "/embeddings.csv");
    ds.queries = read_matrix_csv(input_dir + "/queries.csv");

    std::ifstream ids_in(input_dir + "/ids.csv");
    if (!ids_in.is_open()) {
        throw std::runtime_error("Failed to open ids.csv for read.");
    }
    std::string line;
    while (std::getline(ids_in, line)) {
        if (!line.empty()) {
            ds.ids.push_back(std::stoll(line));
        }
    }
    return ds;
}

}  // namespace vecscale
