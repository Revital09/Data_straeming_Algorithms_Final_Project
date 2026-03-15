from __future__ import annotations

import os
import csv
import itertools
import tracemalloc
from typing import Dict, List, Any

import numpy as np

from results import Algo
from tuned_utils import tuned_algorithms
from utils import set_seed
from data_new2 import make_real_datasets, make_synthetic_datasets


OUTPUT_DIR = "results_new"
SWEEP_ID = "Kmeans_sweep"

STREAM_MODEL = "insertion-only"

SEEDS = [42, 77, 211]

SYNTHETIC_SETUPS = [
    {"n": 10_000, "d": 10, "k": 8},
    {"n": 30_000, "d": 25, "k": 16},
]

SYNTHETIC_DATASET_NAMES = ["blobs", "anisotropic", "high_dim_sparseish"]
REAL_DATASET_NAMES = ["real_iris", "real_covertype", "real_mnist_pca50"]

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def safe_float(x: Any) -> float:
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")

def is_number(v: Any) -> bool:
    try:
        float(v)
        return True
    except Exception:
        return False

def trunc_ndecimals(x: float, decimals: int) -> float:
    f = 10 ** decimals
    return float(np.trunc(x * f) / f)

def trunc_any(v: Any, decimals: int = 5) -> Any:
    if v is None:
        return v
    try:
        x = float(v)
        if not np.isfinite(x):
            return x
        return trunc_ndecimals(x, decimals)
    except Exception:
        return v

def truncate_numeric_in_rows(rows: List[Dict[str, Any]], decimals: int = 5) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        rr: Dict[str, Any] = {}
        for k, v in r.items():
            if isinstance(v, (int, float, np.integer, np.floating)) or (isinstance(v, str) and is_number(v)):
                rr[k] = trunc_any(v, decimals)
            else:
                rr[k] = v
        out.append(rr)
    return out

def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def run_with_measurements(algo: Algo, X: np.ndarray, y: np.ndarray | None, k: int, rng: np.random.Generator):

    tracemalloc.start()
    try:
        res = algo.fit(X, k, rng, y)
        current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    if res.extra is None:
        res.extra = {}

    # store peak memory bytes (python allocations)
    res.extra["peak_mem_bytes"] = int(peak)

    # ensure points_seen exists for throughput
    if getattr(res, "points_seen", None) is None:
        res.points_seen = int(res.extra.get("points_seen", X.shape[0]))
    if "points_seen" not in res.extra:
        res.extra["points_seen"] = int(res.points_seen)

    # throughput points/sec
    rt = float(res.runtime_sec) if res.runtime_sec is not None else float("nan")
    if np.isfinite(rt) and rt > 0:
        res.extra["throughput_pts_per_sec"] = float(res.points_seen) / rt
    else:
        res.extra["throughput_pts_per_sec"] = float("nan")
        print("error throuput is negative or inf")

    return res


# EXPERIMENT CORE
def run_one_dataset_once(X: np.ndarray, y: np.ndarray | None, k: int, seed: int, algorithms: Algo) -> Dict[str, Any]:
    rng = set_seed(seed)

    # baseline
    kmeans = run_with_measurements(algorithms[0], X, y, k, rng)
    kmeans.cost_ratio_vs_kmeans = 1.0
    kmeans_cost = kmeans.cost_sse

    results = {algorithms[0].name: kmeans}
    for algo in algorithms[1:]:
        res = run_with_measurements(algo, X, y, k, rng)
        res.cost_ratio_vs_kmeans = res.cost_sse / (kmeans_cost + 1e-12)
        results[algo.name] = res
        print(f"Completed algo={algo.name} runtime_sec={res.runtime_sec:.5f}, peak_mem_bytes={res.extra['peak_mem_bytes']}")

    return results


def flatten_result(
    sweep_id: str,
    dataset_name: str,
    n: int,
    d: int,
    k: int,
    seed: int,
    algo_name: str,
    res
) -> Dict[str, Any]:
    ratio = safe_float(res.cost_ratio_vs_kmeans)
    qloss_pct = (ratio - 1.0) * 100.0 if np.isfinite(ratio) else float("nan")

    row = {
        "sweep_id": sweep_id,
        "dataset": dataset_name,
        "n": n,
        "d": d,
        "k": k,
        "seed": seed,
        "algorithm": algo_name,
        "stream_model": STREAM_MODEL,  
        "runtime_sec": safe_float(res.runtime_sec),
        "memory": safe_float(res.memory),
        "cost_sse": safe_float(res.cost_sse),
        "cost_ratio_vs_kmeans": ratio,
        "quality_loss_pct_vs_kmeans": qloss_pct,
        "ari": safe_float(res.ari),
        "nmi": safe_float(res.nmi),
    }
    if getattr(res, "extra", None):
        for ek, ev in res.extra.items():
            row[f"extra__{ek}"] = ev
    return row


def main():
    ensure_dir(OUTPUT_DIR)

    raw_rows: List[Dict[str, Any]] = []
    print("tuning the parameters of the algorithms")

    algorithms = tuned_algorithms()
    print(f"num algorithms = {len(algorithms)}")
    for algo in algorithms:
        print(f" - {algo.name}")

    # synthetic 
    for setup in SYNTHETIC_SETUPS:
        n = setup["n"]
        d = setup["d"]
        k = setup["k"]

        rng_gen = set_seed(1234 + 17 * k + 3 * d + (n % 1000))
        datasets = make_synthetic_datasets(rng_gen, n=n, d=d, k_true=k)

        synthetic_datasets = {
            name: datasets[name]
            for name in SYNTHETIC_DATASET_NAMES
            if name in datasets
        }

        for dataset_name, (X, y) in synthetic_datasets.items():
            for seed in SEEDS:
                results = run_one_dataset_once(X, y, k=k, seed=seed, algorithms=algorithms)

                for algo_name, res in results.items():
                    raw_rows.append(
                        flatten_result(
                            sweep_id=SWEEP_ID,
                            dataset_name=dataset_name,
                            n=n,
                            d=d,
                            k=k,
                            seed=seed,
                            algo_name=algo_name,
                            res=res,
                        )
                    )

            print(f"Completed synthetic dataset={dataset_name} for n={n}, d={d}, k={k}")

    # real
    rng_real = set_seed(999)
    datasets = make_real_datasets(rng_real)

    real_datasets = {
        name: datasets[name]
        for name in REAL_DATASET_NAMES
        if name in datasets
    }

    for dataset_name, (X, y) in real_datasets.items():
        n_real = int(X.shape[0])
        d_real = int(X.shape[1])
        k_real = int(len(np.unique(y)))

        for seed in SEEDS:
            results = run_one_dataset_once(X, y, k=k_real, seed=seed, algorithms=algorithms)
            for algo_name, res in results.items():
                raw_rows.append(
                    flatten_result(
                        sweep_id=SWEEP_ID,
                        dataset_name=dataset_name,
                        n=n_real,
                        d=d_real,
                        k=k_real,
                        seed=seed,
                        algo_name=algo_name,
                        res=res
                    )
                )
        print(f"\nCompleted real dataset={dataset_name} with true n={n_real}, d={d_real}, k={k_real}")

    raw_rows_out = truncate_numeric_in_rows(raw_rows, decimals=5)
    raw_csv = os.path.join(OUTPUT_DIR, f"{SWEEP_ID}_raw.csv")
    write_csv(raw_csv, raw_rows_out)

    print("Done.")
    print(f"Raw CSV: {raw_csv}")
if __name__ == "__main__":
    main()
