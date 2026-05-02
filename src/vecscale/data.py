from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def save_dataset(output_path: Path, embeddings: np.ndarray, ids: np.ndarray, queries: np.ndarray) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        embeddings=embeddings.astype(np.float32),
        ids=ids.astype(np.int64),
        queries=queries.astype(np.float32),
    )


def load_dataset(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path)
    return {
        "embeddings": data["embeddings"].astype(np.float32, copy=False),
        "ids": data["ids"].astype(np.int64, copy=False),
        "queries": data["queries"].astype(np.float32, copy=False),
    }


def create_shards(dataset_path: Path, shard_dir: Path, num_shards: int) -> List[Path]:
    shard_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(dataset_path)
    embeddings = dataset["embeddings"]
    ids = dataset["ids"]

    splits = np.array_split(np.arange(embeddings.shape[0]), num_shards)
    shard_paths: List[Path] = []

    for shard_id, split in enumerate(splits):
        shard_path = shard_dir / f"shard_{shard_id:03d}.npz"
        np.savez_compressed(
            shard_path,
            embeddings=embeddings[split].astype(np.float32),
            ids=ids[split].astype(np.int64),
        )
        shard_paths.append(shard_path)

    metadata = {
        "dataset_path": str(dataset_path),
        "num_shards": num_shards,
        "shards": [str(p) for p in shard_paths],
        "total_vectors": int(embeddings.shape[0]),
        "dimension": int(embeddings.shape[1]),
    }
    (shard_dir / "shards.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return shard_paths
