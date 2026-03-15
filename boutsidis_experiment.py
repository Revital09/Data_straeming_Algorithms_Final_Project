from __future__ import annotations

import json
import math
import os
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

from boutsidis_streaming import (
    _rademacher_projection_matrix,
    Boutsidis_Streaming,
)


# Helpers
def choose_r(d: int, k: int, eps: float, c2: float, r_min: int = 10) -> int:
    algo = Boutsidis_Streaming(eps=eps, c2=c2, r_min=r_min)
    return algo._choose_r(d, k)


def sample_pairs(n: int, n_pairs: int, rng: np.random.Generator) -> np.ndarray:
    """
    Returns array of shape (m, 2) with sampled index pairs (i, j), i != j.
    """
    if n < 2:
        raise ValueError("Need at least 2 points to sample pairs.")

    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)

    same = (i == j)
    while np.any(same):
        j[same] = rng.integers(0, n, size=np.sum(same))
        same = (i == j)

    return np.column_stack([i, j])


def pairwise_distortion_ratios(
    X: np.ndarray,
    XR: np.ndarray,
    pairs: np.ndarray,
    tiny: float = 1e-12,
) -> np.ndarray:
    """
    ratio = ||Rx_i - Rx_j||^2 / ||x_i - x_j||^2
    """
    xi = X[pairs[:, 0]]
    xj = X[pairs[:, 1]]
    xri = XR[pairs[:, 0]]
    xrj = XR[pairs[:, 1]]

    orig_d2 = np.sum((xi - xj) ** 2, axis=1)
    proj_d2 = np.sum((xri - xrj) ** 2, axis=1)

    valid = orig_d2 > tiny
    ratios = proj_d2[valid] / orig_d2[valid]
    return ratios


# Experiment A: Rademacher sanity check
def experiment_rademacher_distribution(
    d: int,
    r: int,
    seeds: Tuple[int, ...],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    rows = []

    target_abs = 1.0 / math.sqrt(r)

    for seed in seeds:
        rng = np.random.default_rng(seed)
        R = _rademacher_projection_matrix(d=d, r=r, rng=rng)

        vals, counts = np.unique(R, return_counts=True)
        total = R.size

        pos_val = target_abs
        neg_val = -target_abs

        pos_count = int(np.sum(np.isclose(R, pos_val)))
        neg_count = int(np.sum(np.isclose(R, neg_val)))
        other_count = int(total - pos_count - neg_count)

        rows.append(
            {
                "seed": int(seed),
                "d": int(d),
                "r": int(r),
                "total_entries": int(total),
                "positive_fraction": float(pos_count / total),
                "negative_fraction": float(neg_count / total),
                "other_fraction": float(other_count / total),
                "matrix_mean": float(np.mean(R)),
                "matrix_std": float(np.std(R)),
                "num_unique_values": int(len(vals)),
                "unique_values": [float(v) for v in vals.tolist()],
                "unique_values_ok": bool(
                    len(vals) == 2 and
                    np.all(np.isclose(np.sort(vals), np.array([neg_val, pos_val]), atol=1e-7))
                ),
            }
        )

    df = pd.DataFrame(rows)

    summary = {
        "mean_positive_fraction": float(df["positive_fraction"].mean()),
        "std_positive_fraction": float(df["positive_fraction"].std(ddof=0)),
        "mean_negative_fraction": float(df["negative_fraction"].mean()),
        "std_negative_fraction": float(df["negative_fraction"].std(ddof=0)),
        "mean_other_fraction": float(df["other_fraction"].mean()),
        "mean_matrix_mean": float(df["matrix_mean"].mean()),
        "std_matrix_mean": float(df["matrix_mean"].std(ddof=0)),
        "mean_matrix_std": float(df["matrix_std"].mean()),
        "std_matrix_std": float(df["matrix_std"].std(ddof=0)),
        "all_unique_values_ok": bool(df["unique_values_ok"].all()),
        "target_abs_value": float(target_abs),
    }

    return summary, rows


# Experiment B: JL distance preservation
def experiment_jl_distance_preservation(
    X: np.ndarray,
    k: int,
    eps: float,
    c2: float,
    r_min: int,
    seeds: Tuple[int, ...],
    n_pairs: int = 5000,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], np.ndarray, int]:
    X = np.asarray(X, dtype=np.float32)
    n, d = X.shape
    r = choose_r(d=d, k=k, eps=eps, c2=c2, r_min=r_min)

    rows = []
    all_ratios = []

    for seed in seeds:
        rng = np.random.default_rng(seed)

        R = _rademacher_projection_matrix(d=d, r=r, rng=rng)
        XR = X @ R
        pairs = sample_pairs(n=n, n_pairs=n_pairs, rng=rng)
        ratios = pairwise_distortion_ratios(X, XR, pairs)

        within_eps = np.mean((ratios >= 1.0 - eps) & (ratios <= 1.0 + eps))
        within_2eps = np.mean((ratios >= 1.0 - 2 * eps) & (ratios <= 1.0 + 2 * eps))

        rows.append(
            {
                "seed": int(seed),
                "d": int(d),
                "r": int(r),
                "n_pairs": int(len(ratios)),
                "mean_ratio": float(np.mean(ratios)),
                "std_ratio": float(np.std(ratios)),
                "min_ratio": float(np.min(ratios)),
                "max_ratio": float(np.max(ratios)),
                "median_ratio": float(np.median(ratios)),
                "within_eps_fraction": float(within_eps),
                "within_2eps_fraction": float(within_2eps),
            }
        )

        all_ratios.append(ratios)

    all_ratios_concat = np.concatenate(all_ratios, axis=0)
    df = pd.DataFrame(rows)

    summary = {
        "mean_ratio": float(df["mean_ratio"].mean()),
        "std_of_mean_ratio": float(df["mean_ratio"].std(ddof=0)),
        "mean_std_ratio": float(df["std_ratio"].mean()),
        "mean_min_ratio": float(df["min_ratio"].mean()),
        "mean_max_ratio": float(df["max_ratio"].mean()),
        "mean_median_ratio": float(df["median_ratio"].mean()),
        "mean_within_eps_fraction": float(df["within_eps_fraction"].mean()),
        "std_within_eps_fraction": float(df["within_eps_fraction"].std(ddof=0)),
        "mean_within_2eps_fraction": float(df["within_2eps_fraction"].mean()),
        "std_within_2eps_fraction": float(df["within_2eps_fraction"].std(ddof=0)),
    }

    return summary, rows, all_ratios_concat, r


def save_histogram(ratios: np.ndarray, eps: float, output_path: str) -> None:
    plt.figure(figsize=(8, 6))
    plt.hist(ratios, bins=50)
    plt.axvline(1.0 - eps, linestyle="--")
    plt.axvline(1.0 + eps, linestyle="--")
    plt.axvline(1.0, linestyle="-")
    plt.xlabel("Distance distortion ratio")
    plt.ylabel("Frequency")
    plt.title("JL distance preservation: projected/original squared distance ratio")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# Main runner
def run_boutsidis_assumptions(
    X: np.ndarray,
    k: int,
    eps: float,
    c2: float,
    r_min: int,
    output_dir: str,
    seeds: Tuple[int, ...] = (42, 77, 211),
    n_pairs: int = 5000,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)

    X = np.asarray(X, dtype=np.float32)
    n, d = X.shape
    r = choose_r(d=d, k=k, eps=eps, c2=c2, r_min=r_min)

    rad_summary, rad_rows = experiment_rademacher_distribution(
        d=d,
        r=r,
        seeds=seeds,
    )

    jl_summary, jl_rows, ratios, r_used = experiment_jl_distance_preservation(
        X=X,
        k=k,
        eps=eps,
        c2=c2,
        r_min=r_min,
        seeds=seeds,
        n_pairs=n_pairs,
    )

    pd.DataFrame(rad_rows).to_csv(
        os.path.join(output_dir, "rademacher_by_seed.csv"),
        index=False
    )

    pd.DataFrame(jl_rows).to_csv(
        os.path.join(output_dir, "jl_preservation_by_seed.csv"),
        index=False
    )

    save_histogram(
        ratios=ratios,
        eps=eps,
        output_path=os.path.join(output_dir, "jl_ratio_hist.png"),
    )

    results = {
        "config": {
            "n_samples": int(n),
            "d": int(d),
            "k": int(k),
            "eps": float(eps),
            "c2": float(c2),
            "r_min": int(r_min),
            "r": int(r_used),
            "n_pairs": int(n_pairs),
            "seeds": [int(s) for s in seeds],
        },
        "experiment_rademacher": rad_summary,
        "experiment_jl": jl_summary,
    }

    with open(os.path.join(output_dir, "assumptions_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    X, y = make_blobs(
        n_samples=10000,
        centers=8,
        n_features=20,
        cluster_std=2.0,
        random_state=42,
    )
    X = X.astype(np.float32)

    results = run_boutsidis_assumptions(
        X=X,
        k=8,
        eps=0.8,
        c2=8.0,
        r_min=2,
        output_dir="output_assumptions/boutsidis_assumptions",
        seeds=(42, 77, 211),
        n_pairs=5000,
    )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()