from __future__ import annotations
import time
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from results import Algo, Result
from utils import kmeans_cost_sse, assign_labels

class KMeansAlgo(Algo):
    name = "KMeans(sk)"

    def __init__(self, max_iter: int = 300):
        self.max_iter = max_iter

    def fit(self, X: np.ndarray, k: int, rng: np.random.Generator, y=None) -> Result:
        t0 = time.perf_counter()
        km = KMeans(n_clusters=k, max_iter=self.max_iter,
                    random_state=int(rng.integers(1, 1_000_000)))
        km.fit(X)
        t1 = time.perf_counter()

        centers = km.cluster_centers_
        cost = kmeans_cost_sse(X, centers)

        labels = assign_labels(X, centers)
        ari = adjusted_rand_score(y, labels) if y is not None else None
        nmi = normalized_mutual_info_score(y, labels) if y is not None else None
        state_bytes = int(X.nbytes + centers.nbytes + km.labels_.nbytes)

        return Result(
            centers=centers,
            runtime_sec=t1 - t0,
            memory=float(state_bytes),
            cost_sse=cost,
            cost_ratio_vs_kmeans=1.0,
            ari=ari,
            nmi=nmi,
        )
