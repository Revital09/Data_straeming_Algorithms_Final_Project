from __future__ import annotations
import numpy as np
import pandas as pd
from results import Result


def extract_quality(result: Result) -> float:
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


def minmax_normalize(s: pd.Series) -> pd.Series:
    """
    Normalize series to [0,1]
    """
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
    Rank parameter combinations by a weighted tradeoff score.

    Higher quality is better.
    Lower runtime is better.
    Lower memory is better.
    """

    df = agg.copy()

    df["quality_norm"] = minmax_normalize(df[quality_col])
    df["runtime_norm"] = minmax_normalize(df[runtime_col])
    df["memory_norm"] = minmax_normalize(df[memory_col])

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