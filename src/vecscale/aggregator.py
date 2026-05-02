from __future__ import annotations

from typing import List, Tuple

import numpy as np


ShardResult = Tuple[np.ndarray, np.ndarray]


def merge_topk(shard_results: List[ShardResult], top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Merge shard-local top-K results into global top-K."""
    if not shard_results:
        raise ValueError("No shard results provided for merge.")

    all_ids = np.concatenate([item[0] for item in shard_results], axis=1)
    all_scores = np.concatenate([item[1] for item in shard_results], axis=1)

    top_k = min(top_k, all_scores.shape[1])
    idx = np.argpartition(-all_scores, kth=top_k - 1, axis=1)[:, :top_k]
    scores = np.take_along_axis(all_scores, idx, axis=1)
    ids = np.take_along_axis(all_ids, idx, axis=1)

    order = np.argsort(-scores, axis=1)
    merged_scores = np.take_along_axis(scores, order, axis=1)
    merged_ids = np.take_along_axis(ids, order, axis=1)
    return merged_ids, merged_scores
