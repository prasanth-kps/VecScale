#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate synthetic embedding/query dataset.")
    parser.add_argument("--output", type=str, required=True, help="Output .npz dataset path.")
    parser.add_argument("--num-vectors", type=int, default=100000, help="Number of corpus vectors.")
    parser.add_argument("--num-queries", type=int, default=1000, help="Number of query vectors.")
    parser.add_argument("--dim", type=int, default=128, help="Embedding dimension.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    rng = np.random.default_rng(args.seed)

    embeddings = rng.standard_normal((args.num_vectors, args.dim), dtype=np.float32)
    queries = rng.standard_normal((args.num_queries, args.dim), dtype=np.float32)
    ids = np.arange(args.num_vectors, dtype=np.int64)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        embeddings=embeddings,
        ids=ids,
        queries=queries,
    )
    print(f"Dataset written to {output_path}")
    print(f"embeddings={embeddings.shape}, queries={queries.shape}")


if __name__ == "__main__":
    main()