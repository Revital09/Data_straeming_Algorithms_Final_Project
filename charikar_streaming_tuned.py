import json
import os

import matplotlib
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

from charikar_streaming import Charikar_KMeans
from results import Result

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _extract_quality(result: Result) -> float:
    """
    Higher is better.
    Priority:
    1) NMI
    2) ARI
    3) negative SSE
    """
    if result.nmi is not None:
        return float(result.nmi)
    if result.ari is not None:
        return float(result.ari)
    return -float(result.cost_sse)


def _minmax_normalize(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    mn = s.min()
    mx = s.max()
    if abs(mx - mn) < 1e-12:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mn) / (mx - mn)


def pick_best_overall(
    agg: pd.DataFrame,
    quality_col: str = "quality_mean",
    runtime_col: str = "runtime_sec_mean",
    memory_col: str = "memory_mean",
    quality_weight: float = 0.5,
    runtime_weight: float = 0.25,
    memory_weight: float = 0.25,
):
    """
    Rank all parameter combinations by one weighted tradeoff score.

    Higher quality is better.
    Lower runtime is better.
    Lower memory is better.

    Returns:
        scored_df: all combinations with normalized metrics and tradeoff_score
        best_one_df: a one-row dataframe with the best overall combination
    """
    df = agg.copy()

    df["quality_norm"] = _minmax_normalize(df[quality_col])
    df["runtime_norm"] = _minmax_normalize(df[runtime_col])
    df["memory_norm"] = _minmax_normalize(df[memory_col])

    df["tradeoff_score"] = (
        quality_weight * df["quality_norm"]
        - runtime_weight * df["runtime_norm"]
        - memory_weight * df["memory_norm"]
    )

    df = df.sort_values(
        by=["tradeoff_score", quality_col, runtime_col, memory_col],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)

    best_one = df.head(1).copy()
    return df, best_one


def tune_charikar_parameters(
    samples: np.ndarray,
    k: int,
    output_dir: str,
    labels=None,
    chunk_size: int = 4092,
    beta_values=(5.0, 10.0, 25.0),
    gamma_values=(20.0, 50.0, 100.0),
    seeds=(42, 77, 211),
    quality_weight: float = 0.5,
    runtime_weight: float = 0.25,
    memory_weight: float = 0.25,
):
    """
    Runs Charikar_KMeans for every (beta, gamma) combination, aggregates metrics
    across seeds, ranks them by one overall score, saves CSVs and graphs, and
    returns the single best overall combination.

    Saved files in output_dir:
      - charikar_all_results.csv
      - charikar_aggregated_results.csv
      - charikar_scored_results.csv
      - charikar_best_overall.csv
      - best_overall.json
      - memory_vs_quality.png
      - runtime_vs_quality.png
    """
    os.makedirs(output_dir, exist_ok=True)

    rows = []

    for beta in beta_values:
        for gamma in gamma_values:
            for seed in seeds:
                rng = np.random.default_rng(seed)

                algo = Charikar_KMeans(
                    beta=beta,
                    gamma=gamma,
                    chunk_size=chunk_size,
                )

                result = algo.fit(samples, k=k, rng=rng, y=labels)
                quality = _extract_quality(result)
                extra = result.extra or {}

                rows.append(
                    {
                        "seed": int(seed),
                        "chunk_size": int(chunk_size),
                        "beta": float(beta),
                        "gamma": float(gamma),
                        "runtime_sec": float(result.runtime_sec),
                        "memory": float(result.memory),
                        "cost_sse": float(result.cost_sse),
                        "ari": None if result.ari is None else float(result.ari),
                        "nmi": None if result.nmi is None else float(result.nmi),
                        "quality": float(quality),
                        "points_seen": int(extra.get("points_seen", -1)),
                        "dimension": int(extra.get("dimension", -1)),
                        "num_phases": int(extra.get("num_phases", -1)),
                        "final_summary_size": int(extra.get("final_summary_size", -1)),
                        "final_lower_bound": float(extra.get("final_lower_bound", np.nan)),
                        "avg_update_ms": float(extra.get("avg_update_ms", np.nan)),
                    }
                )
            print(f"Finished beta={beta}, gamma={gamma}")

    df_all = pd.DataFrame(rows)
    df_all.to_csv(os.path.join(output_dir, "charikar_all_results.csv"), index=False)

    agg = (
        df_all.groupby(["chunk_size", "beta", "gamma"], as_index=False)
        .agg(
            runtime_sec_mean=("runtime_sec", "mean"),
            runtime_sec_std=("runtime_sec", "std"),
            memory_mean=("memory", "mean"),
            memory_std=("memory", "std"),
            cost_sse_mean=("cost_sse", "mean"),
            cost_sse_std=("cost_sse", "std"),
            quality_mean=("quality", "mean"),
            quality_std=("quality", "std"),
            num_phases_mean=("num_phases", "mean"),
            final_summary_size_mean=("final_summary_size", "mean"),
            final_lower_bound_mean=("final_lower_bound", "mean"),
            avg_update_ms_mean=("avg_update_ms", "mean"),
            avg_update_ms_std=("avg_update_ms", "std"),
            points_seen_mean=("points_seen", "mean"),
            ari_mean=("ari", "mean"),
            nmi_mean=("nmi", "mean"),
        )
        .reset_index(drop=True)
    )

    agg.to_csv(os.path.join(output_dir, "charikar_aggregated_results.csv"), index=False)

    scored_df, best_one_df = pick_best_overall(
        agg=agg,
        quality_col="quality_mean",
        runtime_col="runtime_sec_mean",
        memory_col="memory_mean",
        quality_weight=quality_weight,
        runtime_weight=runtime_weight,
        memory_weight=memory_weight,
    )

    scored_df.to_csv(os.path.join(output_dir, "charikar_scored_results.csv"), index=False)
    best_one_df.to_csv(os.path.join(output_dir, "charikar_best_overall.csv"), index=False)

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
        f"BEST beta={best_row['beta']}, gamma={best_row['gamma']}",
        (best_row["memory_mean"], best_row["quality_mean"]),
    )
    plt.xlabel("Memory usage (bytes, mean)")
    plt.ylabel("Quality (mean)")
    plt.title("Charikar tuning: Memory usage vs Quality")
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
        f"BEST beta={best_row['beta']}, gamma={best_row['gamma']}",
        (best_row["runtime_sec_mean"], best_row["quality_mean"]),
    )
    plt.xlabel("Runtime (sec, mean)")
    plt.ylabel("Quality (mean)")
    plt.title("Charikar tuning: Runtime vs Quality")
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

    best_df = tune_charikar_parameters(
        samples=X,
        k=8,
        output_dir="output/charikar_blobs",
        labels=y,
        chunk_size=4092,
        beta_values=(3.0, 5.0, 25.0),
        gamma_values=(10.0, 30.0, 100.0),
        seeds=(42, 77, 211),
        quality_weight=0.5,
        runtime_weight=0.25,
        memory_weight=0.25,
    )

    print("Best overall parameter combination:")
    print(best_df)


if __name__ == "__main__":
    main()
