import json
import os

import matplotlib
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

from boutsidis_streaming import Boutsidis_Streaming
from utils import extract_quality, pick_best_overall

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def tune_boutsidis_parameters(
    samples: np.ndarray,
    k: int,
    r_min: int,
    output_dir: str,
    labels=None,
    chunk_size: int = 1024,
    eps_values=(0.3, 0.5, 0.8),
    c2_values=( 8.0, 12.0, 16.0),
    seeds=(42, 77, 211),
    quality_weight: float = 0.5,
    runtime_weight: float = 0.25,
    memory_weight: float = 0.25,
):
    """
    Runs Boutsidis_Streaming for every (eps, c2) combination, aggregates metrics
    across seeds, ranks them by one overall score, saves CSVs and graphs, and
    returns the single best overall combination.

    Saved files in output_dir:
      - boutsidis_all_results.csv
      - boutsidis_aggregated_results.csv
      - boutsidis_scored_results.csv
      - boutsidis_best_overall.csv
      - best_overall.json
      - memory_vs_quality.png
      - runtime_vs_quality.png
    """
    os.makedirs(output_dir, exist_ok=True)

    rows = []

    for eps in eps_values:
        for c2 in c2_values:
            for seed in seeds:
                rng = np.random.default_rng(seed)

                algo = Boutsidis_Streaming(
                    eps=eps,
                    c2=c2,
                    chunk_size=chunk_size,
                    r_min=r_min
                )

                result = algo.fit(samples=samples, k=k, rng=rng, labels=labels)
                quality = extract_quality(result)
                extra = result.extra or {}

                rows.append(
                    {
                        "seed": int(seed),
                        "chunk_size": int(chunk_size),
                        "eps": float(eps),
                        "c2": float(c2),
                        "runtime_sec": float(result.runtime_sec),
                        "memory": float(result.memory),
                        "cost_sse": float(result.cost_sse),
                        "ari": None if result.ari is None else float(result.ari),
                        "nmi": None if result.nmi is None else float(result.nmi),
                        "quality": float(quality),
                        "r": int(extra.get("r", -1)),
                        "d": int(extra.get("d", -1)),
                        "points_seen": int(extra.get("points_seen", -1)),
                        "avg_update_ms": float(extra.get("avg_update_ms", np.nan)),
                        "cost_is_approx": bool(extra.get("cost_is_approx", False)),
                        "stream_is_single_pass": bool(extra.get("stream_is_single_pass", False)),
                    }
                )
            print(f"Completed eps={eps}, c2={c2}")

    df_all = pd.DataFrame(rows)
    df_all.to_csv(os.path.join(output_dir, "boutsidis_all_results.csv"), index=False)

    agg = (
        df_all.groupby(["chunk_size", "eps", "c2"], as_index=False)
        .agg(
            runtime_sec_mean=("runtime_sec", "mean"),
            runtime_sec_std=("runtime_sec", "std"),
            memory_mean=("memory", "mean"),
            memory_std=("memory", "std"),
            cost_sse_mean=("cost_sse", "mean"),
            cost_sse_std=("cost_sse", "std"),
            quality_mean=("quality", "mean"),
            quality_std=("quality", "std"),
            r_mean=("r", "mean"),
            r_std=("r", "std"),
            avg_update_ms_mean=("avg_update_ms", "mean"),
            avg_update_ms_std=("avg_update_ms", "std"),
            points_seen_mean=("points_seen", "mean"),
            ari_mean=("ari", "mean"),
            nmi_mean=("nmi", "mean"),
        )
        .reset_index(drop=True)
    )

    agg.to_csv(os.path.join(output_dir, "boutsidis_aggregated_results.csv"), index=False)

    scored_df, best_one_df = pick_best_overall(
        agg=agg,
        quality_col="quality_mean",
        runtime_col="runtime_sec_mean",
        memory_col="memory_mean",
        quality_weight=quality_weight,
        runtime_weight=runtime_weight,
        memory_weight=memory_weight,
    )

    scored_df.to_csv(os.path.join(output_dir, "boutsidis_scored_results.csv"), index=False)
    best_one_df.to_csv(os.path.join(output_dir, "boutsidis_best_overall.csv"), index=False)

    with open(os.path.join(output_dir, "best_overall.json"), "w", encoding="utf-8") as f:
        json.dump(best_one_df.to_dict(orient="records"), f, indent=2)

    best_row = best_one_df.iloc[0]

    plt.figure(figsize=(8, 6))
    plt.scatter(scored_df["memory_mean"], scored_df["quality_mean"])
    plt.scatter(
        [best_row["memory_mean"]],
        [best_row["quality_mean"]],
        marker="x",
        s=120,
    )
    plt.annotate(
        f"BEST eps={best_row['eps']}, c2={best_row['c2']}",
        (best_row["memory_mean"], best_row["quality_mean"]),
    )
    plt.xlabel("Memory usage (bytes, mean)")
    plt.ylabel("Quality (mean)")
    plt.title("Boutsidis tuning: Memory usage vs Quality")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "memory_vs_quality.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(scored_df["runtime_sec_mean"], scored_df["quality_mean"])
    plt.scatter(
        [best_row["runtime_sec_mean"]],
        [best_row["quality_mean"]],
        marker="x",
        s=120,
    )
    plt.annotate(
        f"BEST eps={best_row['eps']}, c2={best_row['c2']}",
        (best_row["runtime_sec_mean"], best_row["quality_mean"]),
    )
    plt.xlabel("Runtime (sec, mean)")
    plt.ylabel("Quality (mean)")
    plt.title("Boutsidis tuning: Runtime vs Quality")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "runtime_vs_quality.png"), dpi=150)
    plt.close()

    return best_one_df


def main():
    X, y = make_blobs(
        n_samples=10000,
        centers=8,
        n_features=20,
        cluster_std=2.0,
        random_state=42,
    )

    X = X.astype("float32")

    best_df = tune_boutsidis_parameters(
        samples=X,
        k=8,
        r_min=2,
        output_dir="output/boutsidis_blobs",
        labels=y,
        eps_values = (1.5, 2.5, 3.5),
        c2_values = (1.0, 2.0, 3.0),
        seeds=(42, 77, 211),
        quality_weight=0.5,
        runtime_weight=0.25,
        memory_weight=0.25,
    )

    print("Best overall parameter combination:")
    print(best_df)


if __name__ == "__main__":
    main()
