import json
import os
import numpy as np
from sklearn.datasets import make_blobs

from ailon_streaming import Ailon_Coreset


def run_flat(algo, X, k, rng, labels=None):
    return algo.fit(X, k=k, rng=rng, labels=labels)


def merge_two_summaries(algo, X1, w1, X2, w2, k, rng):
    X_merge = np.vstack([X1, X2])
    w_merge = np.hstack([w1, w2])
    sq_norms_merge = algo._squared_norms(X_merge)

    n_total = X_merge.shape[0]
    coreset_size = max(1, int(np.ceil(algo.coreset_factor * np.log(max(k, 2)))))
    reps = max(1, int(np.ceil(algo.repeat_factor * np.log(max(n_total, 2)))))

    centers = algo._calculate_centers(
        samples=X_merge,
        sample_sq_norms=sq_norms_merge,
        w=w_merge,
        k=k,
        rng=rng,
        coreset_size=coreset_size,
        reps=reps,
    )

    X_red, w_red = algo._induce_summary(
        samples=X_merge,
        sample_sq_norms=sq_norms_merge,
        w=w_merge,
        centers=centers,
    )
    return X_red, w_red


def run_merge_reduce(algo, X, k, rng, labels=None):
    summaries = []

    for start in range(0, X.shape[0], algo.chunk_size):
        block = X[start:start + algo.chunk_size]
        block_sq_norms = algo._squared_norms(block)
        w_block = np.ones(block.shape[0], dtype=float)

        coreset_size = max(1, int(np.ceil(algo.coreset_factor * np.log(max(k, 2)))))
        reps = max(1, int(np.ceil(algo.repeat_factor * np.log(max(block.shape[0], 2)))))

        centers = algo._calculate_centers(
            samples=block,
            sample_sq_norms=block_sq_norms,
            w=w_block,
            k=k,
            rng=rng,
            coreset_size=coreset_size,
            reps=reps,
        )

        pts, wts = algo._induce_summary(
            samples=block,
            sample_sq_norms=block_sq_norms,
            w=w_block,
            centers=centers,
        )
        summaries.append((pts, wts))

    # two-stage merge-reduce:
    reduced = []
    i = 0
    while i < len(summaries):
        if i + 1 < len(summaries):
            X_red, w_red = merge_two_summaries(
                algo,
                summaries[i][0], summaries[i][1],
                summaries[i + 1][0], summaries[i + 1][1],
                k,
                rng,
            )
            reduced.append((X_red, w_red))
            i += 2
        else:
            reduced.append(summaries[i])
            i += 1

    summary_X = np.vstack([s[0] for s in reduced])
    summary_w = np.hstack([s[1] for s in reduced])

    centers_final = algo._kmeanspp_seed(summary_X, summary_w, k, rng)

    pred, cost = algo._assign_and_cost(X, centers_final)

    ari = None
    nmi = None
    if labels is not None:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        ari = adjusted_rand_score(labels, pred)
        nmi = normalized_mutual_info_score(labels, pred)

    memory = int(summary_X.nbytes + summary_w.nbytes)

    from results import Result
    return Result(
        centers=centers_final,
        runtime_sec=np.nan,
        memory=memory,
        cost_sse=cost,
        cost_ratio_vs_kmeans=np.nan,
        ari=ari,
        nmi=nmi,
        extra={
            "summary_size": int(summary_X.shape[0]),
            "algorithm": "Ailon two-stage merge-reduce"
        }
    )


def main():
    X, y = make_blobs(
        n_samples=10000,
        centers=8,
        n_features=10,
        cluster_std=2.0,
        random_state=42
    )
    X = X.astype(np.float32)

    algo = Ailon_Coreset(
        chunk_size=1024,
        coreset_factor=1.0,
        repeat_factor=0.75
    )

    flat_rng = np.random.default_rng(42)
    flat_res = run_flat(algo, X, k=8, rng=flat_rng, labels=y)

    mr_rng = np.random.default_rng(42)
    mr_res = run_merge_reduce(algo, X, k=8, rng=mr_rng, labels=y)

    print("=== Flat summary ===")
    print("memory:", flat_res.memory)
    print("cost_sse:", flat_res.cost_sse)
    print("ari:", flat_res.ari)
    print("nmi:", flat_res.nmi)
    print("summary_size:", flat_res.extra["summary_size"])

    print("\n=== Merge-reduce ===")
    print("memory:", mr_res.memory)
    print("cost_sse:", mr_res.cost_sse)
    print("ari:", mr_res.ari)
    print("nmi:", mr_res.nmi)
    print("summary_size:", mr_res.extra["summary_size"])

    output_dir = "output_assumptions/ailon_assumption"
    os.makedirs(output_dir, exist_ok=True)

    results = {
        "flat_summary": {
            "memory": float(flat_res.memory),
            "cost_sse": float(flat_res.cost_sse),
            "ari": None if flat_res.ari is None else float(flat_res.ari),
            "nmi": None if flat_res.nmi is None else float(flat_res.nmi),
            "summary_size": int(flat_res.extra.get("summary_size", -1))
        },
        "merge_reduce": {
            "memory": float(mr_res.memory),
            "cost_sse": float(mr_res.cost_sse),
            "ari": None if mr_res.ari is None else float(mr_res.ari),
            "nmi": None if mr_res.nmi is None else float(mr_res.nmi),
            "summary_size": int(mr_res.extra.get("summary_size", -1))
        }
    }

    json_path = os.path.join(output_dir, "merge_reduce_assumption_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {json_path}")


if __name__ == "__main__":
    main()