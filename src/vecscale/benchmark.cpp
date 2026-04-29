#include "vecscale/benchmark.hpp"
#include <algorithm>
#include <chrono>
namespace vecscale {
BenchmarkSummary run_benchmark(QueryRouter& router, const Matrix& queries, std::size_t top_k){
    BenchmarkSummary s{}; s.query_count=queries.size(); if(queries.empty()) return s;
    std::vector<double> l; l.reserve(queries.size()); auto t0=std::chrono::high_resolution_clock::now();
    for(const auto& q: queries){ Matrix one{q}; auto a=std::chrono::high_resolution_clock::now(); (void)router.search(one, top_k); auto b=std::chrono::high_resolution_clock::now(); l.push_back(std::chrono::duration<double,std::milli>(b-a).count()); }
    auto t1=std::chrono::high_resolution_clock::now(); std::sort(l.begin(), l.end()); std::chrono::duration<double> total=t1-t0;
    s.throughput_qps = total.count()>0.0? static_cast<double>(queries.size())/total.count():0.0; s.p50_ms=l[l.size()*50/100]; s.p95_ms=l[l.size()*95/100]; return s;
}
}
