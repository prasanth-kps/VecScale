#!/usr/bin/env python3
from __future__ import annotations

import argparse
import numpy as np


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run VecScale scaffold demo.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset .npz.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    dataset = np.load(args.dataset)
    print("Day 1 scaffold demo")
    print(f"embeddings shape: {dataset['embeddings'].shape}")
    print(f"queries shape: {dataset['queries'].shape}")
    print("Distributed benchmark integration will be added in later commits.")


if __name__ == "__main__":
    main()