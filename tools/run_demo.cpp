/// run_demo — load a config file and execute the full VecScale benchmark pipeline.
///
/// Usage:
///   run_demo [--config <path>]   (default: configs/default.json)

#include "vecscale/benchmark.hpp"
#include <iostream>
#include <string>
#include <stdexcept>
#include <iomanip>

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [--config <path>]\n";
}

int main(int argc, char* argv[]) {
    std::string config_path = "configs/default.json";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--config" || arg == "-c") && i+1 < argc)
            config_path = argv[++i];
        else { print_usage(argv[0]); return 1; }
    }

    try {
        vecscale::BenchmarkConfig cfg = vecscale::load_config(config_path);
        vecscale::BenchmarkResult res = vecscale::run_benchmark(cfg);

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "=== Final Summary ===\n"
                  << "recall@"   << cfg.top_k << "      : " << res.recall_at_k     << "\n"
                  << "dist QPS        : " << static_cast<int>(res.distributed_qps) << "\n"
                  << "baseline QPS    : " << static_cast<int>(res.baseline_qps)    << "\n"
                  << "p50 latency ms  : " << res.p50_latency_ms                    << "\n"
                  << "p99 latency ms  : " << res.p99_latency_ms                    << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
