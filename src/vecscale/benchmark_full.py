from __future__ import annotations

import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from multiprocessing import Process
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from .baselines import exact_baseline_topk, measure_single_node_baseline
from .data import create_shards, load_dataset
from .router import QueryRouter, WorkerEndpoint
from .worker import run_worker_server


@dataclass
class BenchmarkConfig:
    dataset_path: str
    output_dir: str = "results"
    shard_dir: str = "data/shards"
    host: str = "127.0.0.1"
    base_port: int = 55000
    num_shards: int = 4
    top_k: int = 10
    batch_size: int = 8
    concurrency_levels: Sequence[int] = (1, 4, 8)
    warmup_queries: int = 10
    use_gpu: bool = False
    auth_key: str = "vecscale"


def _batched(arr: np.ndarray, batch_size: int):
    for start in range(0, len(arr), batch_size):
        yield arr[start : start + batch_size]


def _recall_at_k(pred_ids: np.ndarray, gt_ids: np.ndarray, k: int) -> float:
    k = min(k, pred_ids.shape[1], gt_ids.shape[1])
    pred = pred_ids[:, :k]
    gt = gt_ids[:, :k]
    recalls = []
    for i in range(pred.shape[0]):
        pred_set = set(pred[i].tolist())
        gt_set = set(gt[i].tolist())
        recalls.append(len(pred_set.intersection(gt_set)) / max(1, len(gt_set)))
    return float(np.mean(recalls))


def _launch_workers(config: BenchmarkConfig, shard_paths: List[Path]) -> List[Process]:
    workers: List[Process] = []
    for worker_id, shard_path in enumerate(shard_paths):
        process = Process(
            target=run_worker_server,
            args=(
                worker_id,
                str(shard_path),
                config.host,
                config.base_port + worker_id,
                config.auth_key,
                config.use_gpu,
            ),
            daemon=True,
        )
        process.start()
        workers.append(process)
    time.sleep(0.5)
    return workers


def _build_router(config: BenchmarkConfig) -> QueryRouter:
    endpoints = [
        WorkerEndpoint(worker_id=i, host=config.host, port=config.base_port + i)
        for i in range(config.num_shards)
    ]
    return QueryRouter(endpoints=endpoints, auth_key=config.auth_key)


def _run_concurrency_pass(
    router: QueryRouter,
    queries: np.ndarray,
    top_k: int,
    batch_size: int,
    concurrency: int,
) -> Dict:
    latencies_ms: List[float] = []
    predicted_ids: List[np.ndarray] = []
    communication_ms: List[float] = []
    merge_ms: List[float] = []

    def _execute_query_batch(batch: np.ndarray):
        start = time.perf_counter()
        ids, _, meta = router.search(batch, top_k=top_k)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return ids, elapsed_ms, meta

    start_total = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_execute_query_batch, batch) for batch in _batched(queries, batch_size)]
        for future in as_completed(futures):
            ids, elapsed_ms, meta = future.result()
            predicted_ids.append(ids)
            latencies_ms.extend([elapsed_ms] * len(ids))
            communication_ms.append(meta["scatter_gather_ms"])
            merge_ms.append(meta["merge_ms"])
    elapsed_total = time.perf_counter() - start_total

    pred_ids = np.concatenate(predicted_ids, axis=0) if predicted_ids else np.empty((0, top_k), dtype=np.int64)
    return {
        "pred_ids": pred_ids,
        "latencies_ms": latencies_ms,
        "elapsed_s": elapsed_total,
        "throughput_qps": float(len(queries) / elapsed_total) if elapsed_total > 0 else 0.0,
        "p50_ms": float(np.percentile(latencies_ms, 50)) if latencies_ms else 0.0,
        "p95_ms": float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0,
        "avg_scatter_gather_ms": float(np.mean(communication_ms)) if communication_ms else 0.0,
        "avg_merge_ms": float(np.mean(merge_ms)) if merge_ms else 0.0,
    }


def run_benchmark(config: BenchmarkConfig) -> Dict:
    dataset_path = Path(config.dataset_path)
    output_dir = Path(config.output_dir)
    shard_dir = Path(config.shard_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(dataset_path)
    embeddings = dataset["embeddings"]
    ids = dataset["ids"]
    queries = dataset["queries"]

    gt_ids, _ = exact_baseline_topk(
        queries=queries,
        embeddings=embeddings,
        ids=ids,
        top_k=config.top_k,
        use_gpu=config.use_gpu,
    )
    baseline_metrics = measure_single_node_baseline(
        queries=queries,
        embeddings=embeddings,
        ids=ids,
        top_k=config.top_k,
        use_gpu=config.use_gpu,
    )

    shard_paths = create_shards(dataset_path=dataset_path, shard_dir=shard_dir, num_shards=config.num_shards)
    workers = _launch_workers(config, shard_paths)
    router = _build_router(config)

    for warm_batch in _batched(queries[: config.warmup_queries], config.batch_size):
        router.search(warm_batch, top_k=config.top_k)

    runs = []
    latency_rows = []

    try:
        for concurrency in config.concurrency_levels:
            metrics = _run_concurrency_pass(
                router=router,
                queries=queries,
                top_k=config.top_k,
                batch_size=config.batch_size,
                concurrency=int(concurrency),
            )
            recall = _recall_at_k(metrics["pred_ids"], gt_ids, k=config.top_k)
            run_summary = {
                "concurrency": int(concurrency),
                "throughput_qps": metrics["throughput_qps"],
                "p50_ms": metrics["p50_ms"],
                "p95_ms": metrics["p95_ms"],
                "recall_at_k": recall,
                "avg_scatter_gather_ms": metrics["avg_scatter_gather_ms"],
                "avg_merge_ms": metrics["avg_merge_ms"],
            }
            runs.append(run_summary)

            for lat in metrics["latencies_ms"]:
                latency_rows.append(
                    {
                        "concurrency": int(concurrency),
                        "latency_ms": float(lat),
                    }
                )
    finally:
        router.shutdown_workers()
        for process in workers:
            process.join(timeout=1.0)
            if process.is_alive():
                process.kill()

    latency_log_path = output_dir / "latency_log.csv"
    with latency_log_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["concurrency", "latency_ms"])
        writer.writeheader()
        writer.writerows(latency_rows)

    summary = {
        "config": asdict(config),
        "baseline": baseline_metrics,
        "distributed_runs": runs,
        "artifacts": {
            "latency_log_csv": str(latency_log_path),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["artifacts"]["summary_json"] = str(summary_path)
    return summary
