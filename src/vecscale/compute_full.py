from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - torch is optional at runtime
    torch = None


def l2_normalize(vectors: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, eps)


def _topk_numpy(sim_matrix: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    k = min(k, sim_matrix.shape[1])
    idx = np.argpartition(-sim_matrix, kth=k - 1, axis=1)[:, :k]
    scores = np.take_along_axis(sim_matrix, idx, axis=1)
    order = np.argsort(-scores, axis=1)
    sorted_idx = np.take_along_axis(idx, order, axis=1)
    sorted_scores = np.take_along_axis(scores, order, axis=1)
    return sorted_scores, sorted_idx


def _topk_torch(sim_matrix: "torch.Tensor", k: int) -> Tuple[np.ndarray, np.ndarray]:
    k = min(k, sim_matrix.shape[1])
    scores, indices = torch.topk(sim_matrix, k=k, dim=1, largest=True, sorted=True)
    return scores.cpu().numpy(), indices.cpu().numpy()


def topk_cosine_similarity(
    queries: np.ndarray,
    vectors: np.ndarray,
    top_k: int,
    use_gpu: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (scores, local_indices) with shape [batch_size, top_k]."""
    queries = queries.astype(np.float32, copy=False)
    vectors = vectors.astype(np.float32, copy=False)

    q_norm = l2_normalize(queries)
    v_norm = l2_normalize(vectors)

    if use_gpu and torch is not None and torch.cuda.is_available():
        q_t = torch.from_numpy(q_norm).cuda(non_blocking=True)
        v_t = torch.from_numpy(v_norm).cuda(non_blocking=True)
        sim = q_t @ v_t.T
        return _topk_torch(sim, top_k)

    sim = q_norm @ v_norm.T
    return _topk_numpy(sim, top_k)
