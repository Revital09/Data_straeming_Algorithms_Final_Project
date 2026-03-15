from __future__ import annotations
import time
import math
import numpy as np

from results import Algo, Result


class Ailon_Coreset(Algo):
    """
    Ailon, Jaiswal, Monteleoni (2009)-style streaming k-means#
    with tunable summary-size and repetition factors.

    Parameters
    ----------
    chunk_size : int
        Number of raw points processed per block.

    coreset_factor : float
        Controls L = ceil(coreset_factor * log(k)).
        Larger => more representatives per block => more memory, usually better quality.

    repeat_factor : float
        Controls reps = ceil(repeat_factor * log(n_total)).
        Larger => more repeated k-means# trials => more runtime, usually better quality.
    """
    name = "[Ailon2009] Streaming k-means"

    def __init__(
        self,
        chunk_size: int = 1024,
        coreset_factor: float = 3.0,
        repeat_factor: float = 3.0
    ):
        self.chunk_size = chunk_size
        self.coreset_factor = coreset_factor
        self.repeat_factor = repeat_factor

    @staticmethod
    def _squared_norms(X: np.ndarray) -> np.ndarray:
        return np.einsum("ij,ij->i", X, X, optimize=True)

    @staticmethod
    def _squared_distances(
        samples: np.ndarray,
        sample_sq_norms: np.ndarray,
        centers: np.ndarray,
    ) -> np.ndarray:
        center_sq_norms = np.einsum("ij,ij->i", centers, centers, optimize=True)
        dist2 = sample_sq_norms[:, None] + center_sq_norms[None, :]
        dist2 -= 2.0 * (samples @ centers.T)
        np.maximum(dist2, 0.0, out=dist2)
        return dist2

    def _min_squared_distances(
        self,
        samples: np.ndarray,
        sample_sq_norms: np.ndarray,
        centers: np.ndarray,
    ) -> np.ndarray:
        dist2 = self._squared_distances(samples, sample_sq_norms, centers)
        return np.min(dist2, axis=1)

    def _assign_and_cost(
        self,
        samples: np.ndarray,
        centers: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        sample_sq_norms = self._squared_norms(samples)
        dist2 = self._squared_distances(samples, sample_sq_norms, centers)
        assign = np.argmin(dist2, axis=1)
        cost = float(np.sum(dist2[np.arange(samples.shape[0]), assign]))
        return assign, cost

    # k-means++ seeding
    def _kmeanspp_seed(
        self,
        samples: np.ndarray,
        w: np.ndarray,
        k: int,
        rng: np.random.Generator
    ) -> np.ndarray:
        n = samples.shape[0]
        d = samples.shape[1]
        sample_sq_norms = self._squared_norms(samples)
        probs = w / w.sum()
        idx = rng.choice(n, p=probs)
        centers = np.empty((k, d), dtype=samples.dtype)
        centers[0] = samples[idx]
        d2 = self._min_squared_distances(samples, sample_sq_norms, centers[:1])

        for _ in range(1, k):
            probs = w * d2
            s = probs.sum()
            if s <= 1e-12:
                idx = int(rng.integers(0, n))
            else:
                probs = probs / s
                idx = int(rng.choice(n, p=probs))

            centers[_] = samples[idx]
            new_d2 = self._min_squared_distances(samples, sample_sq_norms, centers[_:_ + 1])
            d2 = np.minimum(d2, new_d2)

        return centers

    # #k-means 
    def _kmeans_sharp(
        self,
        samples: np.ndarray,
        sample_sq_norms: np.ndarray,
        w: np.ndarray,
        k: int,
        rng: np.random.Generator,
        coreset_size: int,
    ) -> tuple[np.ndarray, float]:
        n = samples.shape[0]
        d = samples.shape[1]
        total_centers = max(1, k * coreset_size)

        probs = w / w.sum()
        idx = rng.choice(n, size=coreset_size, p=probs)
        centers = np.empty((total_centers, d), dtype=samples.dtype)
        centers[:coreset_size] = samples[idx]
        num_centers = coreset_size
        d2 = self._min_squared_distances(samples, sample_sq_norms, centers[:num_centers])

        for _ in range(k - 1):
            probs = w * d2
            s = probs.sum()
            if s <= 1e-12:
                idx = rng.integers(0, n, size=coreset_size)
            else:
                probs = probs / s
                idx = rng.choice(n, size=coreset_size, p=probs)

            new_centers = samples[idx]
            centers[num_centers:num_centers + coreset_size] = new_centers
            new_d2 = self._min_squared_distances(samples, sample_sq_norms, new_centers)
            d2 = np.minimum(d2, new_d2)
            num_centers += coreset_size

        cost = float(np.dot(w, d2))
        return centers[:num_centers].copy(), cost

    # #repeat k-means and keep best
    def _calculate_centers(
        self,
        samples: np.ndarray,
        sample_sq_norms: np.ndarray,
        w: np.ndarray,
        k: int,
        rng: np.random.Generator,
        coreset_size: int,
        reps: int,
    ) -> np.ndarray:

        best_cost = float("inf")
        best_centers = None

        for _ in range(reps):
            centers, cost = self._kmeans_sharp(
                samples=samples,
                sample_sq_norms=sample_sq_norms,
                w=w,
                k=k,
                rng=rng,
                coreset_size=coreset_size,
            )

            if cost < best_cost:
                best_cost = cost
                best_centers = centers

        return best_centers

    # induce weighted summary
    def _induce_summary(
        self,
        samples: np.ndarray,
        sample_sq_norms: np.ndarray,
        w: np.ndarray,
        centers: np.ndarray
    ):
        d2 = self._squared_distances(samples, sample_sq_norms, centers)
        assign = np.argmin(d2, axis=1)

        new_w = np.zeros(centers.shape[0], dtype=float)
        np.add.at(new_w, assign, w)

        mask = new_w > 0
        return centers[mask], new_w[mask]

    # FIT
    def fit(
        self,
        samples: np.ndarray,
        k: int,
        rng: np.random.Generator,
        labels=None
    ) -> Result:
        t0 = time.perf_counter()
        n = samples.shape[0]
        coreset_size = max(1, math.ceil(self.coreset_factor * math.log(max(k, 2))))
        reps = max(1, math.ceil(self.repeat_factor * math.log(max(n, 2))))

        summary_X = []
        summary_w = []

        chunk_times = []
        points_seen = 0
        max_chunk_summary_size = 0

        for start in range(0, n, self.chunk_size):
            tb0 = time.perf_counter()

            block = samples[start:start + self.chunk_size]
            block_sq_norms = self._squared_norms(block)
            w_block = np.ones(block.shape[0], dtype=float)

            centers = self._calculate_centers(
                samples=block,
                sample_sq_norms=block_sq_norms,
                w=w_block,
                k=k,
                rng=rng,
                coreset_size=coreset_size,
                reps=reps,
            )
            pts, wts = self._induce_summary(
                samples=block,
                sample_sq_norms=block_sq_norms,
                w=w_block,
                centers=centers,
            )

            summary_X.append(pts)
            summary_w.append(wts)

            max_chunk_summary_size = max(max_chunk_summary_size, pts.shape[0])
            points_seen += block.shape[0]

            tb1 = time.perf_counter()
            chunk_times.append(tb1 - tb0)

        summary_X = np.vstack(summary_X)
        summary_w = np.hstack(summary_w)

        centers_final = self._kmeanspp_seed(summary_X, summary_w, k, rng)

        t1 = time.perf_counter()

        pred, cost = self._assign_and_cost(samples, centers_final)

        ari = None
        nmi = None
        if labels is not None:
            from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
            ari = adjusted_rand_score(labels, pred)
            nmi = normalized_mutual_info_score(labels, pred)

        state_bytes = int(summary_X.nbytes + summary_w.nbytes)
        avg_update_ms = float(np.mean(chunk_times) * 1000.0) if chunk_times else float("nan")

        return Result(
            centers=centers_final,
            runtime_sec=t1 - t0,
            memory=state_bytes,
            cost_sse=cost,
            cost_ratio_vs_kmeans=float("nan"),
            ari=ari,
            nmi=nmi,
            extra={
                "summary_size": int(summary_X.shape[0]),
                "max_chunk_summary_size": int(max_chunk_summary_size),
                "chunk_size": int(self.chunk_size),
                "points_seen": int(points_seen),
                "avg_update_ms": float(avg_update_ms),
                "algorithm": "Ailon2009 tuned (no refinement)",
                "coreset_factor": float(self.coreset_factor),
                "repeat_factor": float(self.repeat_factor),
                "coreset_per_round": int(coreset_size),
                "repetitions": int(reps),
            }
        )
