from __future__ import annotations

import json
import os
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from charikar_streaming import Charikar_KMeans, _weighted_kmeans_centers
from utils import assign_labels, kmeans_cost_sse


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def make_default_dataset(
    n: int,
    d: int = 10,
    k: int = 8,
    random_state: int = 321,
) -> tuple[np.ndarray, np.ndarray]:
    X, y = make_blobs(
        n_samples=n,
        centers=k,
        n_features=2,
        cluster_std=1.5,
        random_state=random_state,
    )
    A = np.array([[0.6, -0.8], [0.4, 0.9]], dtype=np.float64)
    X = X @ A

    if d > 2:
        rng = np.random.default_rng(12345 + n + d + k)
        noise = rng.normal(0, 0.1, size=(n, d - 2))
        X = np.hstack([X, noise])

    return X.astype(np.float64), y.astype(np.int32)


def fit_offline_kmeans(
    X: np.ndarray,
    k: int,
    seed: int,
    y: np.ndarray | None = None,
    n_init: int = 10,
    max_iter: int = 300,
) -> dict[str, Any]:
    t0 = time.perf_counter()

    km = KMeans(
        n_clusters=k,
        n_init=n_init,
        max_iter=max_iter,
        random_state=seed,
    )
    pred = km.fit_predict(X)
    centers = km.cluster_centers_

    t1 = time.perf_counter()

    return {
        "centers": centers,
        "pred": pred,
        "cost_sse": float(kmeans_cost_sse(X, centers)),
        "runtime_sec": float(t1 - t0),
        "ari": None if y is None else float(adjusted_rand_score(y, pred)),
        "nmi": None if y is None else float(normalized_mutual_info_score(y, pred)),
    }


def run_charikar_summary(
    X: np.ndarray,
    k: int,
    seed: int,
    beta: float,
    gamma: float,
    chunk_size: int,
    n_init_final: int = 5,
    max_iter_final: int = 300,
    progress_lb_n_init: int = 1,
    progress_lb_max_iter: int = 50,
    progress_lb_approx_factor: float = 2.0,
    max_stalled_phases: int = 8,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)

    algo = Charikar_KMeans(
        beta=beta,
        gamma=gamma,
        chunk_size=chunk_size,
        n_init_final=n_init_final,
        max_iter_final=max_iter_final,
        progress_lb_n_init=progress_lb_n_init,
        progress_lb_max_iter=progress_lb_max_iter,
        progress_lb_approx_factor=progress_lb_approx_factor,
        max_stalled_phases=max_stalled_phases,
    )

    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape

    L = algo._set_lb_kmeans(X, k) / algo.beta

    raw_start_idx = 0
    phase_id = 1
    summary_X = None
    summary_w = None

    while raw_start_idx < n:
        Mi_X, Mi_w, raw_consumed, _ = algo._run_one_phase_chunked(
            X=X,
            raw_start_idx=raw_start_idx,
            summary_X=summary_X,
            summary_w=summary_w,
            Li=L,
            k=k,
            rng=rng,
        )

        summary_X = Mi_X
        summary_w = Mi_w

        next_raw_start = raw_start_idx + raw_consumed
        if raw_consumed > 0:
            raw_start_idx = next_raw_start

        if raw_start_idx < n and summary_X is not None and summary_w is not None:
            progress_lb = algo._phase_progress_lower_bound(
                summary_X=summary_X,
                summary_w=summary_w,
                next_x=X[raw_start_idx],
                k=k,
                rng=rng,
            )
            L = max(algo.beta * L, progress_lb)
        else:
            L *= algo.beta

        if raw_consumed <= 0:
            x = X[raw_start_idx:raw_start_idx + 1]
            w = np.array([1.0], dtype=np.float64)

            if summary_X is None:
                summary_X = x.copy()
                summary_w = w
            else:
                summary_X = np.vstack([summary_X, x])
                summary_w = np.hstack([summary_w, w])

            raw_start_idx += 1

        phase_id += 1

    assert summary_X is not None and summary_w is not None

    summary_X = summary_X.astype(np.float64, copy=False)
    summary_w = summary_w.astype(np.float64, copy=False)

    return {
        "summary_X": summary_X,
        "summary_w": summary_w,
        "summary_size": int(summary_X.shape[0]),
        "memory_bytes": int(summary_X.nbytes + summary_w.nbytes),
        "sum_weights": float(np.sum(summary_w)),
        "num_phases": int(phase_id - 1),
        "n_points": int(n),
        "dimension": int(d),
    }


def final_cluster_from_summary(
    X_full: np.ndarray,
    summary_X: np.ndarray,
    summary_w: np.ndarray,
    k: int,
    seed: int,
    y: np.ndarray | None = None,
    use_weights: bool = True,
    n_init_final: int = 5,
    max_iter_final: int = 300,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)

    if summary_X.shape[0] > k:
        if use_weights:
            centers = _weighted_kmeans_centers(
                summary_X,
                summary_w,
                k=k,
                rng=rng,
                n_init=n_init_final,
                max_iter=max_iter_final,
            )
        else:
            km = KMeans(
                n_clusters=k,
                n_init=n_init_final,
                max_iter=max_iter_final,
                random_state=seed,
            )
            km.fit(summary_X)
            centers = km.cluster_centers_
    elif summary_X.shape[0] == k:
        centers = summary_X.copy()
    else:
        extra_idx = rng.integers(0, summary_X.shape[0], size=k - summary_X.shape[0])
        centers = np.vstack([summary_X, summary_X[extra_idx]])

    pred = assign_labels(X_full, centers)

    return {
        "centers": centers,
        "pred": pred,
        "cost_sse": float(kmeans_cost_sse(X_full, centers)),
        "ari": None if y is None else float(adjusted_rand_score(y, pred)),
        "nmi": None if y is None else float(normalized_mutual_info_score(y, pred)),
    }


def experiment_a_approximation_proxy(
    X: np.ndarray,
    y: np.ndarray | None,
    k: int,
    output_dir: str,
    beta: float,
    gamma: float,
    chunk_size: int,
    seeds: tuple[int, ...] = (42, 77, 211),
    progress_lb_n_init: int = 1,
    progress_lb_max_iter: int = 50,
    progress_lb_approx_factor: float = 2.0,
    max_stalled_phases: int = 8,
) -> pd.DataFrame:
    ensure_dir(output_dir)
    rows = []

    for seed in seeds:
        summary_info = run_charikar_summary(
            X, k, seed, beta, gamma, chunk_size,
            progress_lb_n_init=progress_lb_n_init,
            progress_lb_max_iter=progress_lb_max_iter,
            progress_lb_approx_factor=progress_lb_approx_factor,
            max_stalled_phases=max_stalled_phases,
        )

        charikar_fit = final_cluster_from_summary(
            X_full=X,
            summary_X=summary_info["summary_X"],
            summary_w=summary_info["summary_w"],
            k=k,
            seed=seed,
            y=y,
            use_weights=True,
        )

        offline_fit = fit_offline_kmeans(X, k, seed, y)

        rows.append(
            {
                "seed": seed,
                "charikar_cost_sse": float(charikar_fit["cost_sse"]),
                "offline_kmeans_cost_sse": float(offline_fit["cost_sse"]),
                "cost_ratio_vs_offline_kmeans": float(
                    charikar_fit["cost_sse"] / (offline_fit["cost_sse"] + 1e-12)
                ),
                "charikar_ari": charikar_fit["ari"],
                "charikar_nmi": charikar_fit["nmi"],
                "summary_size": int(summary_info["summary_size"]),
                "memory_bytes": int(summary_info["memory_bytes"]),
                "num_phases": int(summary_info["num_phases"]),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "approximation_proxy.csv"), index=False)
    return df


def experiment_c_weighted_summary_validation(
    X: np.ndarray,
    y: np.ndarray | None,
    k: int,
    output_dir: str,
    beta: float,
    gamma: float,
    chunk_size: int,
    seeds: tuple[int, ...] = (42, 77, 211),
    progress_lb_n_init: int = 1,
    progress_lb_max_iter: int = 50,
    progress_lb_approx_factor: float = 2.0,
    max_stalled_phases: int = 8,
) -> pd.DataFrame:
    ensure_dir(output_dir)
    rows = []
    n = X.shape[0]

    for seed in seeds:
        summary_info = run_charikar_summary(
            X, k, seed, beta, gamma, chunk_size,
            progress_lb_n_init=progress_lb_n_init,
            progress_lb_max_iter=progress_lb_max_iter,
            progress_lb_approx_factor=progress_lb_approx_factor,
            max_stalled_phases=max_stalled_phases,
        )

        weighted_fit = final_cluster_from_summary(
            X_full=X,
            summary_X=summary_info["summary_X"],
            summary_w=summary_info["summary_w"],
            k=k,
            seed=seed,
            y=y,
            use_weights=True,
        )

        unweighted_fit = final_cluster_from_summary(
            X_full=X,
            summary_X=summary_info["summary_X"],
            summary_w=summary_info["summary_w"],
            k=k,
            seed=seed,
            y=y,
            use_weights=False,
        )

        rows.append(
            {
                "seed": seed,
                "summary_size": int(summary_info["summary_size"]),
                "sum_weights": float(summary_info["sum_weights"]),
                "expected_total_mass": float(n),
                "weight_mass_error": float(abs(summary_info["sum_weights"] - n)),
                "weighted_cost_sse": float(weighted_fit["cost_sse"]),
                "unweighted_cost_sse": float(unweighted_fit["cost_sse"]),
                "weighted_vs_unweighted_cost_ratio": float(
                    weighted_fit["cost_sse"] / (unweighted_fit["cost_sse"] + 1e-12)
                ),
                "weighted_ari": weighted_fit["ari"],
                "unweighted_ari": unweighted_fit["ari"],
                "weighted_nmi": weighted_fit["nmi"],
                "unweighted_nmi": unweighted_fit["nmi"],
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "weighted_summary_validation.csv"), index=False)
    return df


def run_all_experiments(
    output_dir: str,
    beta: float = 1.5,
    gamma: float = 0.25,
    chunk_size: int = 4092,
    seeds: tuple[int, ...] = (42, 77, 211),
    base_n: int = 10_000,
    d: int = 10,
    k: int = 8,
    progress_lb_n_init: int = 1,
    progress_lb_max_iter: int = 50,
    progress_lb_approx_factor: float = 2.0,
    max_stalled_phases: int = 8,
) -> dict[str, Any]:
    ensure_dir(output_dir)

    X, y = make_default_dataset(n=base_n, d=d, k=k)

    df_a = experiment_a_approximation_proxy(
        X=X,
        y=y,
        k=k,
        output_dir=os.path.join(output_dir, "approximation_proxy"),
        beta=beta,
        gamma=gamma,
        chunk_size=chunk_size,
        seeds=seeds,
        progress_lb_n_init=progress_lb_n_init,
        progress_lb_max_iter=progress_lb_max_iter,
        progress_lb_approx_factor=progress_lb_approx_factor,
        max_stalled_phases=max_stalled_phases,
    )

    df_c = experiment_c_weighted_summary_validation(
        X=X,
        y=y,
        k=k,
        output_dir=os.path.join(output_dir, "weighted_summary_validation"),
        beta=beta,
        gamma=gamma,
        chunk_size=chunk_size,
        seeds=seeds,
        progress_lb_n_init=progress_lb_n_init,
        progress_lb_max_iter=progress_lb_max_iter,
        progress_lb_approx_factor=progress_lb_approx_factor,
        max_stalled_phases=max_stalled_phases,
    )

    report = {
        "config": {
            "beta": beta,
            "gamma": gamma,
            "chunk_size": chunk_size,
            "seeds": list(seeds),
            "base_n": base_n,
            "d": d,
            "k": k,
            "progress_lb_n_init": progress_lb_n_init,
            "progress_lb_max_iter": progress_lb_max_iter,
            "progress_lb_approx_factor": progress_lb_approx_factor,
            "max_stalled_phases": max_stalled_phases,
        },
        "experiment_A": {
            "mean_cost_ratio_vs_offline_kmeans": float(df_a["cost_ratio_vs_offline_kmeans"].mean()),
            "std_cost_ratio_vs_offline_kmeans": float(df_a["cost_ratio_vs_offline_kmeans"].std(ddof=1)),
        },
        "experiment_C": {
            "mean_weight_mass_error": float(df_c["weight_mass_error"].mean()),
            "mean_weighted_vs_unweighted_cost_ratio": float(df_c["weighted_vs_unweighted_cost_ratio"].mean()),
        },
    }

    with open(os.path.join(output_dir, "charikar_experiment_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def main() -> None:
    report = run_all_experiments(
        output_dir="output_assumptions/charikar_assumption",
        beta=1.5,
        gamma=0.25,
        chunk_size=4092,
        seeds=(42, 77, 211),
        base_n=10_000,
        d=10,
        k=8,
        progress_lb_n_init=1,
        progress_lb_max_iter=50,
        progress_lb_approx_factor=2.0,
        max_stalled_phases=8,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()