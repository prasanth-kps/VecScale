from __future__ import annotations

import time
from typing import Dict, Tuple

import numpy as np

from .compute import topk_cosine_similarity


def exact_baseline_topk(
    queries: np.ndarray,
    embeddings: np.ndarray,
    ids: np.ndarray,
    top_k: int,
    use_gpu: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    scores, local_idx = topk_cosine_similarity(
        queries=queries,
        vectors=embeddings,
        top_k=top_k,
        use_gpu=use_gpu,
    )
    return ids[local_idx], scores


def measure_single_node_baseline(
    queries: np.ndarray,
    embeddings: np.ndarray,
    ids: np.ndarray,
    top_k: int,
    use_gpu: bool = False,
) -> Dict[str, float]:
    start = time.perf_counter()
    exact_baseline_topk(queries, embeddings, ids, top_k=top_k, use_gpu=use_gpu)
    elapsed = time.perf_counter() - start
    throughput = float(len(queries) / elapsed) if elapsed > 0 else 0.0
    return {
        "total_time_s": elapsed,
        "throughput_qps": throughput,
    }
