#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from vecscale.benchmark import BenchmarkConfig, run_benchmark


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run VecScale distributed benchmark demo.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset .npz.")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory for output artifacts.")
    parser.add_argument("--shard-dir", type=str, default="data/shards", help="Directory to write shard files.")
    parser.add_argument("--num-shards", type=int, default=4, help="Number of shard workers.")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K retrieval.")
    parser.add_argument("--batch-size", type=int, default=8, help="Query batch size.")
    parser.add_argument(
        "--concurrency",
        type=str,
        default="1,4,8",
        help="Comma-separated concurrency levels, e.g. 1,4,8",
    )
    parser.add_argument("--use-gpu", action="store_true", help="Enable GPU search if CUDA is available.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    concurrency_levels = tuple(int(x.strip()) for x in args.concurrency.split(",") if x.strip())

    config = BenchmarkConfig(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        shard_dir=args.shard_dir,
        num_shards=args.num_shards,
        top_k=args.top_k,
        batch_size=args.batch_size,
        concurrency_levels=concurrency_levels,
        use_gpu=args.use_gpu,
    )

    summary = run_benchmark(config)

    print("\n=== Benchmark Summary ===")
    print(f"Baseline throughput (single-node exact): {summary['baseline']['throughput_qps']:.2f} qps")
    for run in summary["distributed_runs"]:
        print(
            f"concurrency={run['concurrency']:>2d} | "
            f"throughput={run['throughput_qps']:.2f} qps | "
            f"p50={run['p50_ms']:.2f} ms | "
            f"p95={run['p95_ms']:.2f} ms | "
            f"Recall@K={run['recall_at_k']:.4f}"
        )
    print("\nArtifacts:")
    for key, value in summary["artifacts"].items():
        print(f"- {key}: {value}")
    print("\nRaw JSON:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
