#include "vecscale/data.hpp"
#include <filesystem>
#include <fstream>
#include <random>
namespace vecscale {
Dataset generate_synthetic_dataset(std::size_t nv, std::size_t nq, std::size_t d, unsigned seed){ Dataset ds; ds.embeddings.assign(nv, Vector(d)); ds.queries.assign(nq, Vector(d)); ds.ids.resize(nv); std::mt19937 rng(seed); std::normal_distribution<float> g(0.0f,1.0f); for(std::size_t i=0;i<nv;++i){ ds.ids[i]=static_cast<std::int64_t>(i); for(std::size_t j=0;j<d;++j) ds.embeddings[i][j]=g(rng);} for(std::size_t i=0;i<nq;++i) for(std::size_t j=0;j<d;++j) ds.queries[i][j]=g(rng); return ds; }
void save_dataset_csv(const Dataset& ds, const std::string& out){ std::filesystem::create_directories(out); std::ofstream e(out+"/embeddings.csv"), q(out+"/queries.csv"), id(out+"/ids.csv"); for (auto& r: ds.embeddings){ for(std::size_t i=0;i<r.size();++i){ e<<r[i]<<(i+1<r.size()?",":""); } e<<"\n";} for (auto& r: ds.queries){ for(std::size_t i=0;i<r.size();++i){ q<<r[i]<<(i+1<r.size()?",":""); } q<<"\n";} for(auto v: ds.ids) id<<v<<"\n"; }
Dataset load_dataset_csv(const std::string&) { return {}; }
}
