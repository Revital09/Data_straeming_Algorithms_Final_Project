from __future__ import annotations

import os
import csv
import itertools
import tracemalloc
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd

from results import Algo
from tuned_utils import tuned_algorithms
from utils import set_seed, pick_best_overall
from data_new2 import make_real_datasets, make_synthetic_datasets


OUTPUT_DIR = "results_final"
SWEEP_ID = "Kmeans_sweep"

STREAM_MODEL = "insertion-only"

SEEDS = [42, 77, 211]

SYNTHETIC_SETUPS = [
    {"n": 10_000, "d": 10, "k": 8},
    {"n": 30_000, "d": 25, "k": 16},
]

SYNTHETIC_DATASET_NAMES = ["blobs", "anisotropic", "high_dim_sparseish"]
REAL_DATASET_NAMES = ["real_iris", "real_covertype", "real_mnist_pca50"]


# Match the tuned-algorithm selection rule: prioritize NMI, then runtime and memory.
TRADEOFF_QUALITY_WEIGHT = 0.5
TRADEOFF_RUNTIME_WEIGHT = 0.25
TRADEOFF_MEMORY_WEIGHT = 0.25


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

def aggregate_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Aggregate over seeds per (dataset,n,d,k,algorithm).
    """
    key_fields = ["sweep_id", "dataset", "n", "d", "k", "algorithm"]
    metric_fields = [
        "runtime_sec",
        "memory",
        "cost_sse",
        "cost_ratio_vs_kmeans",
        "quality_loss_pct_vs_kmeans",
        "ari",
        "nmi",
    ]

    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for r in rows:
        key = tuple(r.get(f) for f in key_fields)
        groups.setdefault(key, []).append(r)

    out: List[Dict[str, Any]] = []
    for key, items in groups.items():
        agg = {f: v for f, v in zip(key_fields, key)}
        agg["num_seeds"] = len(items)

        for mf in metric_fields:
            vals = np.array([safe_float(it.get(mf)) for it in items], dtype=np.float64)
            finite = vals[np.isfinite(vals)]
            if finite.size == 0:
                agg[f"{mf}_mean"] = float("nan")
                agg[f"{mf}_std"] = float("nan")
            else:
                agg[f"{mf}_mean"] = float(np.mean(finite))
                agg[f"{mf}_std"] = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0

        out.append(agg)

    return out

def build_summary_overall(rows_raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    One row per algorithm over ALL runs.

    Includes:
      - means/stds for runtime, memory, SSE, ratio, SSE-based quality-loss%, ARI, NMI
      - speedup_vs_kmeans (based on overall kmeans runtime mean)
      - NMI/runtime/memory tradeoff score and rank using the tuned selection strategy
      - tradeoff_quality_loss_pct: distance from the best tradeoff score
    """
    algos = sorted({r["algorithm"] for r in rows_raw})
    per_algo: List[Dict[str, Any]] = []

    def mean_std(arr: np.ndarray) -> Tuple[float, float]:
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return float("nan"), float("nan")
        m = float(np.mean(finite))
        sd = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
        return m, sd

    for algo in algos:
        items = [r for r in rows_raw if r["algorithm"] == algo]

        rt = np.array([safe_float(it.get("runtime_sec")) for it in items], dtype=np.float64)
        mem = np.array([safe_float(it.get("memory")) for it in items], dtype=np.float64)
        sse = np.array([safe_float(it.get("cost_sse")) for it in items], dtype=np.float64)
        ratio = np.array([safe_float(it.get("cost_ratio_vs_kmeans")) for it in items], dtype=np.float64)
        sse_qloss = np.array([safe_float(it.get("quality_loss_pct_vs_kmeans")) for it in items], dtype=np.float64)
        ari = np.array([safe_float(it.get("ari")) for it in items], dtype=np.float64)
        nmi = np.array([safe_float(it.get("nmi")) for it in items], dtype=np.float64)

        rt_m, rt_s = mean_std(rt)
        mem_m, mem_s = mean_std(mem)
        sse_m, sse_s = mean_std(sse)
        ratio_m, ratio_s = mean_std(ratio)
        sse_qloss_m, sse_qloss_s = mean_std(sse_qloss)
        ari_m, ari_s = mean_std(ari)
        nmi_m, nmi_s = mean_std(nmi)

        per_algo.append({
            "algorithm": algo,
            "num_runs": len(items),

            "runtime_sec_mean": rt_m,
            "runtime_sec_std": rt_s,

            "memory_mean": mem_m,
            "memory_std": mem_s,

            "cost_sse_mean": sse_m,
            "cost_sse_std": sse_s,

            "cost_ratio_vs_kmeans_mean": ratio_m,
            "cost_ratio_vs_kmeans_std": ratio_s,

            "quality_loss_pct_mean": sse_qloss_m,
            "quality_loss_pct_std": sse_qloss_s,

            "ari_mean": ari_m,
            "ari_std": ari_s,

            "nmi_mean": nmi_m,
            "nmi_std": nmi_s,
        })

    kmeans_row = next((r for r in per_algo if r["algorithm"] == "KMeans(sk)"), None)
    kmeans_rt = safe_float(kmeans_row["runtime_sec_mean"]) if kmeans_row else float("nan")

    for r in per_algo:
        rt = safe_float(r["runtime_sec_mean"])
        if np.isfinite(kmeans_rt) and np.isfinite(rt) and rt > 0:
            r["speedup_vs_kmeans"] = kmeans_rt / rt
        else:
            r["speedup_vs_kmeans"] = float("nan")

    if per_algo:
        summary_df = pd.DataFrame(per_algo)
        ranked_df, best_one = pick_best_overall(
            summary_df,
            quality_col="nmi_mean",
            runtime_col="runtime_sec_mean",
            memory_col="memory_mean",
            quality_weight=TRADEOFF_QUALITY_WEIGHT,
            runtime_weight=TRADEOFF_RUNTIME_WEIGHT,
            memory_weight=TRADEOFF_MEMORY_WEIGHT,
        )

        best_tradeoff = (
            safe_float(best_one.iloc[0]["tradeoff_score"])
            if not best_one.empty
            else float("nan")
        )
        tradeoff_span = (
            TRADEOFF_QUALITY_WEIGHT
            + TRADEOFF_RUNTIME_WEIGHT
            + TRADEOFF_MEMORY_WEIGHT
        )

        ranked_rows = ranked_df.to_dict("records")
        rank_map = {
            row["algorithm"]: idx + 1
            for idx, row in enumerate(ranked_rows)
            if np.isfinite(safe_float(row.get("tradeoff_score")))
        }
        tradeoff_map = {
            row["algorithm"]: safe_float(row.get("tradeoff_score"))
            for row in ranked_rows
        }

        for r in per_algo:
            algo = r["algorithm"]
            tradeoff = tradeoff_map.get(algo, float("nan"))
            r["tradeoff_score"] = tradeoff
            r["rank"] = rank_map.get(algo, None)

            if np.isfinite(best_tradeoff) and np.isfinite(tradeoff) and tradeoff_span > 0:
                r["tradeoff_quality_loss_pct"] = float(
                    max(0.0, (best_tradeoff - tradeoff) / tradeoff_span * 100.0)
                )
            else:
                r["tradeoff_quality_loss_pct"] = float("nan")

            sp = safe_float(r["speedup_vs_kmeans"])
            nmi_mean = safe_float(r["nmi_mean"])
            mem_mean = safe_float(r["memory_mean"])

            if np.isfinite(tradeoff) and np.isfinite(nmi_mean) and np.isfinite(sp) and np.isfinite(mem_mean):
                r["explanation_en"] = (
                    f"NMI/runtime/memory tradeoff qloss: {r['tradeoff_quality_loss_pct']:.2f}% "
                    f"(0% is best). Mean NMI={nmi_mean:.4f}, speedup vs kmeans={sp:.2f}x, "
                    f"mean memory={mem_mean:.2f}."
                )
            else:
                r["explanation_en"] = "Insufficient data to compute the NMI/runtime/memory tradeoff."

    return per_algo



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
 
    agg_rows = aggregate_rows(raw_rows)
    summary_rows = build_summary_overall(raw_rows)

    # Truncate numeric values to 5 decimals in all CSVs (no rounding)
    raw_rows_out = truncate_numeric_in_rows(raw_rows, decimals=5)
    agg_rows_out = truncate_numeric_in_rows(agg_rows, decimals=5)
    summary_rows_out = truncate_numeric_in_rows(summary_rows, decimals=5)

    raw_csv = os.path.join(OUTPUT_DIR, f"{SWEEP_ID}_raw.csv")
    agg_csv = os.path.join(OUTPUT_DIR, f"{SWEEP_ID}_agg.csv")
    summary_csv = os.path.join(OUTPUT_DIR, f"{SWEEP_ID}_summary_overall.csv")

    write_csv(raw_csv, raw_rows_out)
    write_csv(agg_csv, agg_rows_out)
    write_csv(summary_csv, summary_rows_out)
    

    print("Done.")
    print(f"Raw CSV     : {raw_csv}")
    print(f"Aggregated  : {agg_csv}")
    print(f"Summary     : {summary_csv}")

if __name__ == "__main__":
    main()
