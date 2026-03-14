from __future__ import annotations
import time
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from results import Algo, Result
from utils import assign_labels, kmeans_cost_sse, weighted_kmeans_centers


class Guha_Stream_KMeans(Algo):
    """
    STREAM K-means (Guha et al., ICDE'02) - KMeans instantiation of the STREAM framework:

    1) Process stream in chunks (simulate insertion-only stream).
    2) For each chunk: run KMeans to m summary centers, compute weights per center.
    3) Merge&Reduce tree:
       - Insert summaries into level 0.
       - If occupied, merge two summary-sets and compress back to m (weighted KMeans),
         carry to next level, repeat.
    4) Final: gather remaining summaries and run weighted KMeans down to k.

    We additionally measure:
      - avg_update_ms: average time to process one chunk (including summarization + merges)
      - memory: bytes of final stored summary (Xs, ws)
      - points_seen: number of points processed (should equal n)
    """

    name = "[1]Guha2002_STREAMKMeans_Paper"

    def __init__(self, chunk_size: int = 4096, m_factor: float = 2.0):
        """
        m_factor means: m = ceil(m_factor * k)
        Typical choice in practice is m_factor ~ 2 (i.e., m ~ 2k)
        """
        self.chunk_size = int(chunk_size)
        self.m_factor = float(m_factor)

    @staticmethod
    def _chunk_summarize_kmeans(
        chunk: np.ndarray,
        m: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run KMeans on a chunk and return:
          centers: (m, d)
          weights: (m,) counts per center
        """
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
        w = np.bincount(km.labels_, minlength=m).astype(np.float64)
        return centers, w

    @staticmethod
    def _compress_weighted(
        Xc: np.ndarray,
        wc: np.ndarray,
        m: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compress weighted points (Xc, wc) down to m centers via weighted KMeans,
        then re-aggregate weights by assigning original weighted points to new centers.
        """
        if Xc.shape[0] <= m:
            return Xc, wc

        m = min(m, Xc.shape[0])
        new_centers = weighted_kmeans_centers(Xc, wc, k=m, rng=rng)
        lab = assign_labels(Xc, new_centers)

        new_w = np.zeros(new_centers.shape[0], dtype=np.float64)
        np.add.at(new_w, lab, wc)
        return new_centers, new_w

    def fit(self, X: np.ndarray, k: int, rng: np.random.Generator, y=None) -> Result:
        t0 = time.perf_counter()
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

        # Summary size per chunk/level (m in STREAMKMeans)
        m = max(32, int(np.ceil(self.m_factor * k)))

        # Merge&Reduce buffers: level -> (centers, weights)
        buffers: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        # Streaming metrics
        chunk_times: list[float] = []
        points_seen = 0

        # --- Stream processing ---
        for start in range(0, n, self.chunk_size):
            tb0 = time.perf_counter()

            chunk = X[start : start + self.chunk_size]
            if chunk.shape[0] == 0:
                continue

            # (1) summarize chunk -> m centers + weights
            Xc, wc = self._chunk_summarize_kmeans(chunk=chunk, m=m, rng=rng)

            # (2) insert into level 0 with carry merges
            level = 0
            while True:
                if level not in buffers:
                    buffers[level] = (Xc, wc)
                    break

                X_old, w_old = buffers.pop(level)
                X_merge = np.vstack([X_old, Xc])
                w_merge = np.hstack([w_old, wc])

                # compress back to m
                Xc, wc = self._compress_weighted(X_merge, w_merge, m=m, rng=rng)
                level += 1

            points_seen += chunk.shape[0]
            tb1 = time.perf_counter()
            chunk_times.append(tb1 - tb0)

        # --- Final clustering ---
        all_X = []
        all_w = []
        for (Xi, wi) in buffers.values():
            all_X.append(Xi)
            all_w.append(wi)

        if all_X:
            Xs = np.vstack(all_X).astype(np.float64, copy=False)
            ws = np.hstack(all_w).astype(np.float64, copy=False)
        else:
            # fallback (should rarely happen)
            Xs = X[: min(2000, n)].astype(np.float64, copy=False)
            ws = np.ones(Xs.shape[0], dtype=np.float64)

        centers_final = weighted_kmeans_centers(Xs, ws, k=k, rng=rng)

        t1 = time.perf_counter()
        cost = kmeans_cost_sse(X, centers_final)
        pred = assign_labels(X, centers_final)

        ari = adjusted_rand_score(y, pred) if y is not None else None
        nmi = normalized_mutual_info_score(y, pred) if y is not None else None

        # ✅ memory: final state (summary) bytes
        state_bytes = int(Xs.nbytes + ws.nbytes)

        # ✅ update time per chunk
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
                "summary_points": int(Xs.shape[0]),
                "m_summary": int(m),
                "levels_used": int(len(buffers)),
                "chunk_size": int(self.chunk_size),
                "points_seen": int(points_seen),
                "avg_update_ms": float(avg_update_ms),
                "m_factor": float(self.m_factor),
            },
        )
