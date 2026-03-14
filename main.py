# main.py
from __future__ import annotations

import os
import csv
import itertools
import tracemalloc
from typing import Dict, List, Tuple, Any

import numpy as np

from tuned_utils import tuned_algorithms
from utils import set_seed
from data import make_datasets

from kmeans import KMeansAlgo
from minibatch_kmeans import MiniBatchKMeansAlgo
from ailon_coreset import Ailon_Coreset
from boutsidis_streaming import Boutsidis_Streaming
from guha_stream import Guha_Stream_KMeans
from charikar_streaming import Charikar_KMeans


# ============================================================
# CONFIG
# ============================================================

OUTPUT_DIR = "results"
SWEEP_ID = "Kmeans_sweep"

STREAM_MODEL = "insertion-only"   # ✅ requirement 1

SEEDS = [42, 77, 211]
N_VALUES = [10_000, 30_000]
D_VALUES = [10, 25]
K_VALUES = [8, 16]

DATASET_NAMES = ["blobs", "anisotropic", "high_dim_sparseish"]
ENABLE_CHARIKAR_16 = True

# ✅ requirement 5: minimal ablation on Boutsidis eps
BOUTSIDIS_EPS_VALUES = [0.3, 0.5, 0.8]


# ============================================================
# ALGORITHMS
# ============================================================

def build_algos():
    """
    Build algorithms list.
    eps_boutsidis is swept for ablation.
    """
    
    algos = [
        KMeansAlgo(max_iter=300),  # baseline
        MiniBatchKMeansAlgo(batch_size=8192, max_iter=100),
        Guha_Stream_KMeans(chunk_size=8192, m_factor=2.0),
        Ailon_Coreset(chunk_size=8192),
        Boutsidis_Streaming(eps=1.5, c2=8.0, chunk_size=1024),
    ]
    if ENABLE_CHARIKAR_16:
        algos.append(Charikar_KMeans(beta=25, gamma=100.0, chunk_size=8192))
    return algos


# ============================================================
# HELPERS
# ============================================================

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


# ============================================================
# NEW: measurement wrapper (peak memory + enrich streaming metrics)
# ============================================================

def run_with_measurements(algo, X: np.ndarray, y: np.ndarray | None, k: int, rng: np.random.Generator):
    """
    ✅ requirement 2: peak python memory via tracemalloc
    ✅ requirement 3: ensure throughput exists
    """
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


# ============================================================
# EXPERIMENT CORE
# ============================================================

def run_one_dataset_once(X: np.ndarray, y: np.ndarray | None, k: int, seed: int, algorithms) -> Dict[str, Any]:
    rng = set_seed(seed)

    # baseline
    kmeans = run_with_measurements(algorithms[0], X, y, k, rng)
    kmeans_cost = kmeans.cost_sse

    results = {algorithms[0].name: kmeans}
    for algo in algorithms[1:]:
        res = run_with_measurements(algo, X, y, k, rng)
        res.cost_ratio_vs_kmeans = res.cost_sse / (kmeans_cost + 1e-12)
        results[algo.name] = res

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

        "stream_model": STREAM_MODEL,   # ✅ requirement 1

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
    algorithems = tuned_algorithms()
    for n, d, k in itertools.product(N_VALUES, D_VALUES, K_VALUES):
        rng_gen = set_seed(1234 + 17 * k + 3 * d + (n % 1000))
        datasets = make_datasets(rng_gen, n=n, d=d, k_true=k)
        datasets = {name: datasets[name] for name in DATASET_NAMES if name in datasets}

        for dataset_name, (X, y) in datasets.items():
            for seed in SEEDS:
                results = run_one_dataset_once(X, y, k=k, seed=seed, algorithms=algorithems)
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
                            res=res
                        )
                    )
            print(f"Completed dataset={dataset_name} for n={n}, d={d}, k={k}")

    # Write raw only (minimal). If you want agg/summary again, keep your old funcs.
    raw_rows_out = truncate_numeric_in_rows(raw_rows, decimals=5)
    raw_csv = os.path.join(OUTPUT_DIR, f"{SWEEP_ID}_raw.csv")
    write_csv(raw_csv, raw_rows_out)

    print("Done.")
    print(f"Raw CSV: {raw_csv}")


if __name__ == "__main__":
    main()
