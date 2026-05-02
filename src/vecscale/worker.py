from __future__ import annotations

import time
from multiprocessing.connection import Listener
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from .compute import topk_cosine_similarity


def load_shard(shard_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(shard_path)
    return (
        data["embeddings"].astype(np.float32, copy=False),
        data["ids"].astype(np.int64, copy=False),
    )


def run_worker_server(
    worker_id: int,
    shard_path: str,
    host: str,
    port: int,
    auth_key: str,
    use_gpu: bool = False,
) -> None:
    embeddings, ids = load_shard(Path(shard_path))
    listener = Listener((host, port), authkey=auth_key.encode("utf-8"))

    try:
        while True:
            conn = listener.accept()
            try:
                request: Dict = conn.recv()
                req_type = request.get("type")

                if req_type == "ping":
                    conn.send({"ok": True, "worker_id": worker_id})
                    continue

                if req_type == "shutdown":
                    conn.send({"ok": True, "worker_id": worker_id})
                    break

                if req_type != "search":
                    conn.send({"error": f"Unknown request type: {req_type}"})
                    continue

                queries = request["queries"].astype(np.float32, copy=False)
                top_k = int(request["top_k"])

                start = time.perf_counter()
                scores, local_idx = topk_cosine_similarity(
                    queries=queries,
                    vectors=embeddings,
                    top_k=top_k,
                    use_gpu=use_gpu,
                )
                latency_ms = (time.perf_counter() - start) * 1000.0
                result_ids = ids[local_idx]
                conn.send(
                    {
                        "ids": result_ids,
                        "scores": scores,
                        "worker_id": worker_id,
                        "compute_ms": latency_ms,
                    }
                )
            finally:
                conn.close()
    finally:
        listener.close()
