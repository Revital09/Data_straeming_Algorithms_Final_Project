from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs

from guha_streaming_new import Guha_Stream_KMeans
from utils import extract_quality


def make_synthetic_stream_dataset(
    n: int,
    d: int = 10,
    k_true: int = 8,
    seed: int = 321,
):
    """
    Synthetic dataset for controlled Guha experiments.
    """
    X, y = make_blobs(
        n_samples=n,
        centers=k_true,
        n_features=2,
        cluster_std=1.5,
        random_state=seed,
    )

    A = np.array([[0.6, -0.8], [0.4, 0.9]])
    X = X @ A

    rng = np.random.default_rng(seed)
    if d > 2:
        X = np.hstack([X, rng.normal(0, 0.1, size=(n, d - 2))])

    return X.astype(np.float64), y


def run_single_guha(
    X: np.ndarray,
    y: np.ndarray | None,
    k: int,
    seed: int,
    chunk_size: int,
    m_factor: float,
):
    rng = np.random.default_rng(seed)
    algo = Guha_Stream_KMeans(
        chunk_size=chunk_size,
        m_factor=m_factor,
    )
    result = algo.fit(X, k=k, rng=rng, y=y)
    quality = extract_quality(result)
    extra = result.extra or {}

    return {
        "seed": int(seed),
        "n": int(X.shape[0]),
        "d": int(X.shape[1]),
        "k": int(k),
        "chunk_size": int(chunk_size),
        "m_factor": float(m_factor),
        "runtime_sec": float(result.runtime_sec),
        "memory": float(result.memory),
        "cost_sse": float(result.cost_sse),
        "ari": None if result.ari is None else float(result.ari),
        "nmi": None if result.nmi is None else float(result.nmi),
        "quality": float(quality),
        "summary_points": int(extra.get("summary_points", -1)),
        "m_summary": int(extra.get("m_summary", -1)),
        "levels_used": int(extra.get("levels_used", -1)),
        "points_seen": int(extra.get("points_seen", -1)),
        "avg_update_ms": float(extra.get("avg_update_ms", np.nan)),
    }


def save_summary_and_plots(
    df_all: pd.DataFrame,
    group_cols: list[str],
    output_dir: str,
    prefix: str,
    x_col: str,
):
    os.makedirs(output_dir, exist_ok=True)

    df_all.to_csv(os.path.join(output_dir, f"{prefix}_all_results.csv"), index=False)

    agg = (
        df_all.groupby(group_cols, as_index=False)
        .agg(
            runtime_sec_mean=("runtime_sec", "mean"),
            runtime_sec_std=("runtime_sec", "std"),
            memory_mean=("memory", "mean"),
            memory_std=("memory", "std"),
            cost_sse_mean=("cost_sse", "mean"),
            cost_sse_std=("cost_sse", "std"),
            quality_mean=("quality", "mean"),
            quality_std=("quality", "std"),
            ari_mean=("ari", "mean"),
            ari_std=("ari", "std"),
            nmi_mean=("nmi", "mean"),
            nmi_std=("nmi", "std"),
            summary_points_mean=("summary_points", "mean"),
            summary_points_std=("summary_points", "std"),
            levels_used_mean=("levels_used", "mean"),
            levels_used_std=("levels_used", "std"),
            avg_update_ms_mean=("avg_update_ms", "mean"),
            points_seen_mean=("points_seen", "mean"),
        )
        .reset_index(drop=True)
    )

    agg.to_csv(os.path.join(output_dir, f"{prefix}_aggregated_results.csv"), index=False)

    plt.figure(figsize=(8, 6))
    plt.plot(agg[x_col], agg["memory_mean"], marker="o")
    plt.xlabel(x_col)
    plt.ylabel("Memory usage (bytes, mean)")
    plt.title(f"{prefix}: {x_col} vs Memory")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_{x_col}_vs_memory.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(agg[x_col], agg["runtime_sec_mean"], marker="o")
    plt.xlabel(x_col)
    plt.ylabel("Runtime (sec, mean)")
    plt.title(f"{prefix}: {x_col} vs Runtime")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_{x_col}_vs_runtime.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(agg[x_col], agg["quality_mean"], marker="o")
    plt.xlabel(x_col)
    plt.ylabel("Quality (mean)")
    plt.title(f"{prefix}: {x_col} vs Quality")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_{x_col}_vs_quality.png"), dpi=150)
    plt.close()

    with open(os.path.join(output_dir, f"{prefix}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "prefix": prefix,
                "num_runs": int(len(df_all)),
                "grouped_by": group_cols,
                "x_col": x_col,
            },
            f,
            indent=2,
        )

    return agg


def experiment_stream_size(
    output_dir: str,
    n_values=(5_000, 10_000, 25_000, 50_000, 100_000),
    d: int = 10,
    k_true: int = 8,
    k_algo: int = 8,
    chunk_size: int = 4096,
    m_factor: float = 6.0,
    seeds=(42, 77, 211),
):
    rows = []

    for n in n_values:
        X, y = make_synthetic_stream_dataset(
            n=n,
            d=d,
            k_true=k_true,
            seed=321,
        )

        for seed in seeds:
            row = run_single_guha(
                X=X,
                y=y,
                k=k_algo,
                seed=seed,
                chunk_size=chunk_size,
                m_factor=m_factor,
            )
            rows.append(row)

        print(f"Completed stream-size experiment for n={n}")

    df_all = pd.DataFrame(rows)
    agg = save_summary_and_plots(
        df_all=df_all,
        group_cols=["n"],
        output_dir=output_dir,
        prefix="guha_stream_size",
        x_col="n",
    )
    return df_all, agg


def experiment_m_factor(
    output_dir: str,
    n: int = 50_000,
    d: int = 10,
    k_true: int = 8,
    k_algo: int = 8,
    chunk_size: int = 4096,
    m_factors=(4.0, 6.0, 8.0, 10.0, 12.0),
    seeds=(42, 77, 211),
):
    rows = []

    X, y = make_synthetic_stream_dataset(
        n=n,
        d=d,
        k_true=k_true,
        seed=321,
    )

    for m_factor in m_factors:
        for seed in seeds:
            row = run_single_guha(
                X=X,
                y=y,
                k=k_algo,
                seed=seed,
                chunk_size=chunk_size,
                m_factor=m_factor,
            )
            rows.append(row)

        print(f"Completed m-factor experiment for m_factor={m_factor}")

    df_all = pd.DataFrame(rows)
    agg = save_summary_and_plots(
        df_all=df_all,
        group_cols=["m_factor", "m_summary"],
        output_dir=output_dir,
        prefix="guha_m_factor",
        x_col="m_factor",
    )
    return df_all, agg


def main():
    base_output_dir = "output_algorithms/guha/guha_assumption"
    os.makedirs(base_output_dir, exist_ok=True)

    experiment_stream_size(
        output_dir=os.path.join(base_output_dir, "stream_size"),
        n_values=(5_000, 10_000, 25_000, 50_000, 100_000),
        d=10,
        k_true=8,
        k_algo=8,
        chunk_size=4096,
        m_factor=6.0,
        seeds=(42, 77, 211),
    )

    experiment_m_factor(
        output_dir=os.path.join(base_output_dir, "m_factor"),
        n=50_000,
        d=10,
        k_true=8,
        k_algo=8,
        chunk_size=4096,
        m_factors=(4.0, 6.0, 8.0, 10.0, 12.0),
        seeds=(42, 77, 211),
    )

    print("Guha experiments completed.")


if __name__ == "__main__":
    main()