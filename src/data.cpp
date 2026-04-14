#include "vecscale/data.hpp"
#include <fstream>
#include <stdexcept>
#include <cstring>
#include <filesystem>

namespace vecscale {

// ---- VecScale Data (.vsd) binary format --------------------------------
//  Offset  Size   Field
//  0       8      Magic: "VECSCALE"
//  8       4      Version (uint32, currently 1)
//  12      8      num_vectors (int64)
//  20      8      num_queries (int64)
//  28      4      dim (int32)
//  32      nv*dim*4   embeddings (float32, row-major)
//  +       nv*8       ids (int64)
//  +       nq*dim*4   queries (float32, row-major)
// -------------------------------------------------------------------------

static constexpr char     MAGIC[8] = {'V','E','C','S','C','A','L','E'};
static constexpr uint32_t VERSION  = 1;

void save_dataset(const std::string& path, const Dataset& ds) {
    const auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty())
        std::filesystem::create_directories(parent);

    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot create file: " + path);

    const int64_t nv  = ds.embeddings.rows();
    const int64_t nq  = ds.queries.rows();
    const int32_t dim = static_cast<int32_t>(ds.embeddings.cols());

    f.write(MAGIC, 8);
    f.write(reinterpret_cast<const char*>(&VERSION), sizeof(VERSION));
    f.write(reinterpret_cast<const char*>(&nv),      sizeof(nv));
    f.write(reinterpret_cast<const char*>(&nq),      sizeof(nq));
    f.write(reinterpret_cast<const char*>(&dim),     sizeof(dim));

    // Embeddings are stored row-major (Matrix typedef guarantees this)
    f.write(reinterpret_cast<const char*>(ds.embeddings.data()),
            nv * dim * sizeof(float));
    f.write(reinterpret_cast<const char*>(ds.ids.data()),
            nv * sizeof(int64_t));
    f.write(reinterpret_cast<const char*>(ds.queries.data()),
            nq * dim * sizeof(float));

    if (!f) throw std::runtime_error("Write error: " + path);
}

Dataset load_dataset(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open file: " + path);

    char magic[8];
    f.read(magic, 8);
    if (std::memcmp(magic, MAGIC, 8) != 0)
        throw std::runtime_error("Not a VecScale dataset (.vsd) file: " + path);

    uint32_t version;
    f.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (version != VERSION)
        throw std::runtime_error("Unsupported .vsd version: " + std::to_string(version));

    int64_t nv, nq;
    int32_t dim;
    f.read(reinterpret_cast<char*>(&nv),  sizeof(nv));
    f.read(reinterpret_cast<char*>(&nq),  sizeof(nq));
    f.read(reinterpret_cast<char*>(&dim), sizeof(dim));

    Dataset ds;
    ds.embeddings.resize(nv, dim);
    ds.queries.resize(nq, dim);
    ds.ids.resize(nv);

    f.read(reinterpret_cast<char*>(ds.embeddings.data()),
           nv * dim * sizeof(float));
    f.read(reinterpret_cast<char*>(ds.ids.data()),
           nv * sizeof(int64_t));
    f.read(reinterpret_cast<char*>(ds.queries.data()),
           nq * dim * sizeof(float));

    if (!f) throw std::runtime_error("Truncated or corrupt file: " + path);
    return ds;
}

} // namespace vecscale
