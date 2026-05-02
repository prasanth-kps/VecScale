from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from multiprocessing.connection import Client
from typing import Dict, List, Tuple

import numpy as np

from .aggregator import merge_topk


@dataclass(frozen=True)
class WorkerEndpoint:
    worker_id: int
    host: str
    port: int


class QueryRouter:
    def __init__(self, endpoints: List[WorkerEndpoint], auth_key: str) -> None:
        self.endpoints = endpoints
        self.auth_key = auth_key.encode("utf-8")

    def _query_worker(
        self,
        endpoint: WorkerEndpoint,
        queries: np.ndarray,
        top_k: int,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        start = time.perf_counter()
        conn = Client((endpoint.host, endpoint.port), authkey=self.auth_key)
        try:
            conn.send({"type": "search", "queries": queries, "top_k": top_k})
            response = conn.recv()
        finally:
            conn.close()
        roundtrip_ms = (time.perf_counter() - start) * 1000.0
        return response["ids"], response["scores"], {
            "worker_id": endpoint.worker_id,
            "compute_ms": float(response.get("compute_ms", 0.0)),
            "roundtrip_ms": roundtrip_ms,
        }

    def search(self, queries: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray, Dict]:
        scatter_start = time.perf_counter()
        shard_results = []
        worker_timings = []
        with ThreadPoolExecutor(max_workers=len(self.endpoints)) as pool:
            futures = [
                pool.submit(self._query_worker, endpoint, queries, top_k)
                for endpoint in self.endpoints
            ]
            for future in futures:
                ids, scores, timing = future.result()
                shard_results.append((ids, scores))
                worker_timings.append(timing)
        scatter_gather_ms = (time.perf_counter() - scatter_start) * 1000.0

        merge_start = time.perf_counter()
        merged_ids, merged_scores = merge_topk(shard_results, top_k=top_k)
        merge_ms = (time.perf_counter() - merge_start) * 1000.0

        metadata = {
            "scatter_gather_ms": scatter_gather_ms,
            "merge_ms": merge_ms,
            "worker_timings": worker_timings,
        }
        return merged_ids, merged_scores, metadata

    def shutdown_workers(self) -> None:
        for endpoint in self.endpoints:
            conn = Client((endpoint.host, endpoint.port), authkey=self.auth_key)
            try:
                conn.send({"type": "shutdown"})
                conn.recv()
            finally:
                conn.close()
