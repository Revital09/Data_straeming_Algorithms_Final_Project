import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from guha_streaming import Guha_Stream_KMeans
from utils import extract_quality, pick_best_overall


def tune_guha_parameters(
    samples: np.ndarray,
    k: int,
    output_dir: str,
    labels=None,
    chunk_size: int = 4096,
    m_factor_values=(1.0, 1.5, 2.0, 3.0, 4.0, 5.0),
    seeds=(42, 77, 211),
    quality_weight: float = 0.25,
    runtime_weight: float = 0.25,
    memory_weight: float = 0.25,
    cost_sse_weight: float = 0.25,
):
    os.makedirs(output_dir, exist_ok=True)

    rows = []

    for m_factor in m_factor_values:
        for seed in seeds:
            rng = np.random.default_rng(seed)

            algo = Guha_Stream_KMeans(
                chunk_size=chunk_size,
                m_factor=m_factor,
            )

            result = algo.fit(samples, k=k, rng=rng, y=labels)
            quality = extract_quality(result)
            extra = result.extra or {}

            rows.append(
                {
                    "seed": int(seed),
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
            )
        print(f"Completed m_factor={m_factor}")

    df_all = pd.DataFrame(rows)
    df_all.to_csv(os.path.join(output_dir, "guha_all_results.csv"), index=False)

    agg = (
        df_all.groupby(["chunk_size", "m_factor"], as_index=False)
        .agg(
            runtime_sec_mean=("runtime_sec", "mean"),
            runtime_sec_std=("runtime_sec", "std"),
            memory_mean=("memory", "mean"),
            memory_std=("memory", "std"),
            cost_sse_mean=("cost_sse", "mean"),
            cost_sse_std=("cost_sse", "std"),
            quality_mean=("quality", "mean"),
            quality_std=("quality", "std"),
            summary_points_mean=("summary_points", "mean"),
            summary_points_std=("summary_points", "std"),
            m_summary_mean=("m_summary", "mean"),
            levels_used_mean=("levels_used", "mean"),
            avg_update_ms_mean=("avg_update_ms", "mean"),
            avg_update_ms_std=("avg_update_ms", "std"),
            points_seen_mean=("points_seen", "mean"),
            ari_mean=("ari", "mean"),
            nmi_mean=("nmi", "mean"),
        )
        .reset_index(drop=True)
    )

    agg.to_csv(os.path.join(output_dir, "guha_aggregated_results.csv"), index=False)

    scored_df, best_one_df = pick_best_overall(
        agg=agg,
        quality_col="nmi_mean",
        runtime_col="runtime_sec_mean",
        memory_col="memory_mean",
        cost_sse_col="cost_sse_mean",
        quality_weight=quality_weight,
        runtime_weight=runtime_weight,
        memory_weight=memory_weight,
        cost_sse_weight=cost_sse_weight,
    )

    scored_df.to_csv(os.path.join(output_dir, "guha_scored_results.csv"), index=False)
    best_one_df.to_csv(os.path.join(output_dir, "guha_best_overall.csv"), index=False)

    with open(os.path.join(output_dir, "best_overall.json"), "w", encoding="utf-8") as f:
        json.dump(best_one_df.to_dict(orient="records"), f, indent=2)

    best_row = best_one_df.iloc[0]

    plt.figure(figsize=(8, 6))
    plt.scatter(scored_df["memory_mean"], scored_df["nmi_mean"])
    plt.scatter(
        [best_row["memory_mean"]],
        [best_row["nmi_mean"]],
        marker="x",
        s=120,
    )
    plt.annotate(
        f"BEST m_factor={best_row['m_factor']}",
        (best_row["memory_mean"], best_row["nmi_mean"]),
    )
    plt.xlabel("Memory usage (bytes, mean)")
    plt.ylabel("NMI (mean)")
    plt.title("Guha tuning: Memory usage vs NMI")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "memory_vs_quality.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(scored_df["runtime_sec_mean"], scored_df["nmi_mean"])
    plt.scatter(
        [best_row["runtime_sec_mean"]],
        [best_row["nmi_mean"]],
        marker="x",
        s=120,
    )
    plt.annotate(
        f"BEST m_factor={best_row['m_factor']}",
        (best_row["runtime_sec_mean"], best_row["nmi_mean"]),
    )
    plt.xlabel("Runtime (sec, mean)")
    plt.ylabel("NMI (mean)")
    plt.title("Guha tuning: Runtime vs NMI")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "runtime_vs_quality.png"), dpi=150)
    plt.close()

    return best_one_df


def main():
    d = 10
    n = 10_000
    X, y = make_blobs(n_samples=n, centers=8, n_features=2, cluster_std=1.5, random_state=321)
    A = np.array([[0.6, -0.8], [0.4, 0.9]])
    X = X @ A
    rng = np.random.default_rng(123)
    if d > 2:
        X = np.hstack([X, rng.normal(0, 0.1, size=(n, d - 2))])


    best_df = tune_guha_parameters(
        samples=X,
        k=14,
        output_dir="output_algorithms/guha/guha_tuning",
        labels=y,
        chunk_size=4096,
        m_factor_values=(1.0, 2.0, 3.0, 4.0, 5.0),
        seeds=(42, 77, 211),
        quality_weight=0.25,
        runtime_weight=0.25,
        memory_weight=0.25,
        cost_sse_weight=0.25,
    )

    print("Best overall parameter combination:")
    print(best_df)


if __name__ == "__main__":
    main()
