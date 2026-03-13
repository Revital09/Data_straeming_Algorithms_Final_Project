import json
import os

import matplotlib
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

from boutsidis_streaming import Boutsidis_Streaming
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
                quality = _extract_quality(result)
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
                        "state_bytes": int(extra.get("state_bytes", -1)),
                        "cost_is_approx": bool(extra.get("cost_is_approx", False)),
                        "stream_is_single_pass": bool(extra.get("stream_is_single_pass", False)),
                    }
                )

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
            state_bytes_mean=("state_bytes", "mean"),
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
