import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ailon_streaming import Ailon_Coreset
from sklearn.datasets import make_blobs
from utils import extract_quality, pick_best_overall


def tune_ailon_parameters(
    samples: np.ndarray,
    k: int,
    output_dir: str,
    labels=None,
    chunk_size: int = 1024,
    coreset_factors=(1.0, 2.0, 3.0),
    repeat_factors=(1.0, 2.0, 3.0),
    seeds=(42, 77, 211),
    quality_weight: float = 0.25,
    runtime_weight: float = 0.25,
    memory_weight: float = 0.25,
    cost_sse_weight: float = 0.25,
):
    """
    Runs Ailon_Coreset for every (coreset_factor, repeat_factor) combination,
    aggregates metrics across seeds, ranks them by one overall score,
    saves CSVs and graphs, and returns the single best overall combination.

    Saved files in output_dir:
      - ailon_all_results.csv
      - ailon_aggregated_results.csv
      - ailon_scored_results.csv
      - ailon_best_overall.csv
      - best_overall.json
      - memory_vs_quality.png
      - runtime_vs_quality.png
    """
    os.makedirs(output_dir, exist_ok=True)

    rows = []

    for cf in coreset_factors:
        for rf in repeat_factors:
            for seed in seeds:
                rng = np.random.default_rng(seed)

                algo = Ailon_Coreset(
                    chunk_size=chunk_size,
                    coreset_factor=cf,
                    repeat_factor=rf
                )

                result = algo.fit(samples=samples, k=k, rng=rng, labels=labels)
                quality = extract_quality(result)

                rows.append({
                    "seed": int(seed),
                    "chunk_size": int(chunk_size),
                    "coreset_factor": float(cf),
                    "repeat_factor": float(rf),
                    "runtime_sec": float(result.runtime_sec),
                    "memory": float(result.memory),
                    "cost_sse": float(result.cost_sse),
                    "ari": None if result.ari is None else float(result.ari),
                    "nmi": None if result.nmi is None else float(result.nmi),
                    "quality": float(quality),
                    "summary_size": int(result.extra.get("summary_size", -1)) if result.extra else -1,
                    "max_chunk_summary_size": int(result.extra.get("max_chunk_summary_size", -1)) if result.extra else -1,
                    "avg_update_ms": float(result.extra.get("avg_update_ms", np.nan)) if result.extra else np.nan,
                    "coreset_per_round": int(result.extra.get("coreset_per_round", -1)) if result.extra else -1,
                    "repetitions": int(result.extra.get("repetitions", -1)) if result.extra else -1,
                })
            print(f"Completed coreset_factor={cf}, repeat_factor={rf}")

    df_all = pd.DataFrame(rows)
    df_all.to_csv(os.path.join(output_dir, "ailon_all_results.csv"), index=False)

    agg = (
        df_all
        .groupby(["chunk_size", "coreset_factor", "repeat_factor"], as_index=False)
        .agg(
            runtime_sec_mean=("runtime_sec", "mean"),
            runtime_sec_std=("runtime_sec", "std"),
            memory_mean=("memory", "mean"),
            memory_std=("memory", "std"),
            cost_sse_mean=("cost_sse", "mean"),
            cost_sse_std=("cost_sse", "std"),
            quality_mean=("quality", "mean"),
            quality_std=("quality", "std"),
            summary_size_mean=("summary_size", "mean"),
            summary_size_std=("summary_size", "std"),
            avg_update_ms_mean=("avg_update_ms", "mean"),
            avg_update_ms_std=("avg_update_ms", "std"),
            coreset_per_round_mean=("coreset_per_round", "mean"),
            repetitions_mean=("repetitions", "mean"),
            ari_mean=("ari", "mean"),
            nmi_mean=("nmi", "mean"),
        )
        .reset_index(drop=True)
    )

    agg.to_csv(os.path.join(output_dir, "ailon_aggregated_results.csv"), index=False)

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

    scored_df.to_csv(os.path.join(output_dir, "ailon_scored_results.csv"), index=False)
    best_one_df.to_csv(os.path.join(output_dir, "ailon_best_overall.csv"), index=False)

    with open(os.path.join(output_dir, "best_overall.json"), "w", encoding="utf-8") as f:
        json.dump(best_one_df.to_dict(orient="records"), f, indent=2)

    best_row = best_one_df.iloc[0]

    plt.figure(figsize=(8, 6))
    plt.scatter(scored_df["memory_mean"], scored_df["nmi_mean"])
    plt.scatter([best_row["memory_mean"]], [best_row["nmi_mean"]], marker="x", s=120)
    plt.annotate(
        f"BEST cf={best_row['coreset_factor']}, rf={best_row['repeat_factor']}",
        (best_row["memory_mean"], best_row["nmi_mean"])
    )
    plt.xlabel("Memory usage (mean)")
    plt.ylabel("NMI (mean)")
    plt.title("Ailon tuning: Memory usage vs NMI")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "memory_vs_quality.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(scored_df["runtime_sec_mean"], scored_df["nmi_mean"])
    plt.scatter([best_row["runtime_sec_mean"]], [best_row["nmi_mean"]], marker="x", s=120)
    plt.annotate(
        f"BEST cf={best_row['coreset_factor']}, rf={best_row['repeat_factor']}",
        (best_row["runtime_sec_mean"], best_row["nmi_mean"])
    )
    plt.xlabel("Runtime (sec, mean)")
    plt.ylabel("NMI (mean)")
    plt.title("Ailon tuning: Runtime vs NMI")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "runtime_vs_quality.png"), dpi=150)
    plt.close()

    return best_one_df


def main():
    X, y = make_blobs(
        n_samples=10000,
        centers=8,
        n_features=10,
        cluster_std=2.0,
        random_state=42,
    )

    X = X.astype("float32")

    best_df = tune_ailon_parameters(
        samples=X,
        k=8,
        output_dir="output_algorithms/ailon/ailon_tuning",
        labels=y,
        coreset_factors=(1.0, 1.5, 2.0),
        repeat_factors=(0.75, 1.0, 1.5),
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
