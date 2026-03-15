from __future__ import annotations

import os
import time

import numpy as np


os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from results import Algo, Result


def _weighted_kmeans_centers(
    X: np.ndarray,
    w: np.ndarray,
    k: int,
    rng: np.random.Generator,
    n_init: int = 5,
    max_iter: int = 200,
) -> np.ndarray:
    km = KMeans(
        n_clusters=k,
        n_init=n_init,
        max_iter=max_iter,
        random_state=int(rng.integers(1, 1_000_000)),
    )
    km.fit(X, sample_weight=w)
    return km.cluster_centers_


class Guha_Stream_KMeans(Algo):
    """
    STREAM K-means (Guha et al., ICDE'02) - KMeans instantiation of the STREAM framework.
    """

    name = "[Guha2002] Streaming k-means"

    def __init__(self, chunk_size: int = 4096, m_factor: float = 2.0):
        self.chunk_size = int(chunk_size)
        self.m_factor = float(m_factor)

    @staticmethod
    def _squared_norms(X: np.ndarray) -> np.ndarray:
        return np.einsum("ij,ij->i", X, X, optimize=True)

    @staticmethod
    def _squared_distances(
        X: np.ndarray,
        X_sq_norms: np.ndarray,
        centers: np.ndarray,
    ) -> np.ndarray:
        center_sq_norms = np.einsum("ij,ij->i", centers, centers, optimize=True)
        dist2 = X_sq_norms[:, None] + center_sq_norms[None, :]
        dist2 -= 2.0 * (X @ centers.T)
        np.maximum(dist2, 0.0, out=dist2)
        return dist2

    def _assign_and_cost(self, X: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, float]:
        X_sq_norms = self._squared_norms(X)
        dist2 = self._squared_distances(X, X_sq_norms, centers)
        labels = np.argmin(dist2, axis=1)
        cost = float(np.sum(dist2[np.arange(X.shape[0]), labels]))
        return labels, cost

    @staticmethod
    def _aggregate_weights(labels: np.ndarray, weights: np.ndarray, n_clusters: int) -> np.ndarray:
        new_w = np.zeros(n_clusters, dtype=np.float64)
        np.add.at(new_w, labels, weights)
        return new_w

    @staticmethod
    def _chunk_summarize_kmeans(
        chunk: np.ndarray,
        m: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        m = min(m, chunk.shape[0])
        if m <= 0:
            return (
                np.empty((0, chunk.shape[1]), dtype=np.float64),
                np.empty((0,), dtype=np.float64),
            )

        km = KMeans(
            n_clusters=m,
            random_state=int(rng.integers(1, 1_000_000)),
        )
        km.fit(chunk)

        centers = km.cluster_centers_.astype(np.float64, copy=False)
        w = np.bincount(km.labels_, minlength=m).astype(np.float64, copy=False)
        return centers, w

    def _compress_weighted(
        self,
        Xc: np.ndarray,
        wc: np.ndarray,
        m: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        if Xc.shape[0] <= m:
            return Xc, wc

        m = min(m, Xc.shape[0])
        new_centers = _weighted_kmeans_centers(Xc, wc, k=m, rng=rng)
        labels, _ = self._assign_and_cost(Xc, new_centers)
        new_w = self._aggregate_weights(labels, wc, new_centers.shape[0])
        return new_centers, new_w

    def fit(self, X: np.ndarray, k: int, rng: np.random.Generator, y=None) -> Result:
        t0 = time.perf_counter()

        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape

        if n == 0:
            return Result(
                centers=np.empty((0, d), dtype=np.float64),
                runtime_sec=0.0,
                memory=0.0,
                cost_sse=float("nan"),
                cost_ratio_vs_kmeans=float("nan"),
                ari=None,
                nmi=None,
                extra={
                    "summary_points": 0,
                    "m_summary": 0,
                    "levels_used": 0,
                    "chunk_size": int(self.chunk_size),
                    "points_seen": 0,
                    "avg_update_ms": float("nan"),
                },
            )

        m = max(32, int(np.ceil(self.m_factor * k)))
        buffers: list[tuple[np.ndarray, np.ndarray] | None] = []

        chunk_times: list[float] = []
        points_seen = 0

        for start in range(0, n, self.chunk_size):
            tb0 = time.perf_counter()

            chunk = X[start:start + self.chunk_size]
            if chunk.shape[0] == 0:
                continue

            Xc, wc = self._chunk_summarize_kmeans(chunk=chunk, m=m, rng=rng)

            level = 0
            while True:
                if level == len(buffers):
                    buffers.append((Xc, wc))
                    break

                existing = buffers[level]
                if existing is None:
                    buffers[level] = (Xc, wc)
                    break

                X_old, w_old = existing
                buffers[level] = None
                X_merge = np.vstack((X_old, Xc))
                w_merge = np.hstack((w_old, wc))
                Xc, wc = self._compress_weighted(X_merge, w_merge, m=m, rng=rng)
                level += 1

            points_seen += chunk.shape[0]
            tb1 = time.perf_counter()
            chunk_times.append(tb1 - tb0)

        summaries = [buf for buf in buffers if buf is not None]
        if summaries:
            Xs = np.vstack([Xi for Xi, _ in summaries]).astype(np.float64, copy=False)
            ws = np.hstack([wi for _, wi in summaries]).astype(np.float64, copy=False)
        else:
            Xs = X[: min(2000, n)].astype(np.float64, copy=False)
            ws = np.ones(Xs.shape[0], dtype=np.float64)

        centers_final = _weighted_kmeans_centers(Xs, ws, k=k, rng=rng)
        pred, cost = self._assign_and_cost(X, centers_final)

        ari = adjusted_rand_score(y, pred) if y is not None else None
        nmi = normalized_mutual_info_score(y, pred) if y is not None else None

        state_bytes = int(Xs.nbytes + ws.nbytes)
        avg_update_ms = float(np.mean(chunk_times) * 1000.0) if chunk_times else float("nan")
        levels_used = sum(buf is not None for buf in buffers)

        t1 = time.perf_counter()

        return Result(
            centers=centers_final,
            runtime_sec=t1 - t0,
            memory=float(state_bytes),
            cost_sse=cost,
            cost_ratio_vs_kmeans=float("nan"),
            ari=ari,
            nmi=nmi,
            extra={
                "summary_points": int(Xs.shape[0]),
                "m_summary": int(m),
                "levels_used": int(levels_used),
                "chunk_size": int(self.chunk_size),
                "points_seen": int(points_seen),
                "avg_update_ms": float(avg_update_ms),
                "m_factor": float(self.m_factor),
            },
        )
