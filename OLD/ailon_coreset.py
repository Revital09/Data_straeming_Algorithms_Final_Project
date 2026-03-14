from __future__ import annotations
import time
import math
import numpy as np

from results import Algo, Result
from utils import assign_labels, kmeans_cost_sse


class Ailon_Coreset(Algo):
    """
    Faithful implementation of Ailon, Jaiswal, Monteleoni 2009

    Uses:

    Algorithm 2: k-means#
    Algorithm 3: Streaming divide-and-conquer
    Final step: k-means++ seeding

    NO MiniBatchKMeans
    NO Lloyd iterations
    """
    name = "[Ailon2009] Streaming k-means#"

    def __init__(self, chunk_size: int = 1024):
        self.chunk_size = chunk_size

    # ---------------------------------------------------
    # k-means++ seeding (Algorithm 1)
    # ---------------------------------------------------

    def _kmeanspp_seed(
        self,
        samples: np.ndarray,
        w: np.ndarray,
        k: int,
        rng: np.random.Generator
    ) -> np.ndarray:
        n = samples.shape[0]
        probs = w / w.sum()
        idx = rng.choice(n, p=probs)
        centers = [samples[idx]]
        d2 = np.sum((samples - centers[0]) ** 2, axis=1)

        for _ in range(1, k):
            probs = w * d2
            s = probs.sum()
            if s <= 1e-12:
                idx = int(rng.integers(0, n))
            else:
                probs = probs / s
                idx = int(rng.choice(n, p=probs))

            centers.append(samples[idx])
            new_d2 = np.sum((samples - centers[-1]) ** 2, axis=1)
            d2 = np.minimum(d2, new_d2)

        return np.array(centers)

    # ---------------------------------------------------
    # k-means# (Algorithm 2)
    # ---------------------------------------------------

    def _kmeans_sharp(
        self,
        samples: np.ndarray,
        w: np.ndarray,
        k: int,
        rng: np.random.Generator
    ) -> np.ndarray:
        n = samples.shape[0]
        L = math.ceil(3 * math.log(max(k, 2)))
        probs = w / w.sum()
        idx = rng.choice(n, size=L, p=probs)
        centers = samples[idx].copy()

        for _ in range(k - 1):
            diff = samples[:, None, :] - centers[None, :, :]
            d2 = np.sum(diff * diff, axis=2)
            d2 = np.min(d2, axis=1)
            probs = w * d2
            s = probs.sum()
            if s <= 1e-12:
                idx = rng.integers(0, n, size=L)
            else:
                probs = probs / s
                idx = rng.choice(n, size=L, p=probs)
                
            centers = np.vstack((centers, samples[idx]))

        return centers

    # ---------------------------------------------------
    # repeat k-means# 3 log n times and pick best
    # ---------------------------------------------------

    def  _calculate_centers(
        self,
        samples,
        w,
        k,
        rng,
        n_total
    ):
        reps = math.ceil(3 * math.log(max(n_total, 2)))
        best_cost = float("inf")
        best_centers = None

        for _ in range(reps):
            centers = self._kmeans_sharp(samples, w, k, rng)
            diff = samples[:, None, :] - centers[None, :, :]
            d2 = np.sum(diff * diff, axis=2)
            d2 = np.min(d2, axis=1)
            cost = np.sum(w * d2)
            if cost < best_cost:
                best_cost = cost
                best_centers = centers

        return best_centers

    # ---------------------------------------------------
    # induce weighted summary
    # ---------------------------------------------------

    def _induce_summary(
        self,
        samples,
        w,
        centers
    ):
        diff = samples[:, None, :] - centers[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        assign = np.argmin(d2, axis=1)
        new_w = np.zeros(centers.shape[0])

        for i in range(len(assign)):
            new_w[assign[i]] += w[i]
        mask = new_w > 0

        return centers[mask], new_w[mask]

    # ---------------------------------------------------
    # FIT
    # ---------------------------------------------------

    def fit(self, samples: np.ndarray, k: int, rng: np.random.Generator, labels=None) -> Result:
        t0 = time.perf_counter()
        n = samples.shape[0]
        w_full = np.ones(n)

        summary_X = []
        summary_w = []

        chunk_times = []
        points_seen = 0

        for start in range(0, n, self.chunk_size):
            tb0 = time.perf_counter()

            block = samples[start:start + self.chunk_size]
            w_block = w_full[start:start + self.chunk_size]
            centers = self._calculate_centers(block, w_block, k, rng, n)
            pts, wts = self._induce_summary(block, w_block, centers)

            summary_X.append(pts)
            summary_w.append(wts)

            points_seen += block.shape[0]
            tb1 = time.perf_counter()
            chunk_times.append(tb1 - tb0)

        summary_X = np.vstack(summary_X)
        summary_w = np.hstack(summary_w)

        centers_final = self._kmeanspp_seed(summary_X, summary_w, k, rng)

        t1 = time.perf_counter()
        cost = kmeans_cost_sse(samples, centers_final)
        pred = assign_labels(samples, centers_final)

        ari = None
        nmi = None
        if labels is not None:
            from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
            ari = adjusted_rand_score(labels, pred)
            nmi = normalized_mutual_info_score(labels, pred)

        # ✅ state bytes (final summary)
        state_bytes = int(summary_X.nbytes + summary_w.nbytes)

        avg_update_ms = float(np.mean(chunk_times) * 1000.0) if chunk_times else float("nan")

        return Result(
            centers=centers_final,
            runtime_sec=t1 - t0,
            memory=float(state_bytes),
            cost_sse=cost,
            cost_ratio_vs_kmeans=float("nan"),
            ari=ari,
            nmi=nmi,
            extra={
                "summary_size": int(summary_X.shape[0]),
                "chunk_size": int(self.chunk_size),
                "points_seen": int(points_seen),
                "avg_update_ms": float(avg_update_ms),
                "algorithm": "Ailon2009 faithful (no refinement)"
            }
        )
