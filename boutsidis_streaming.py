from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Optional, Tuple

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from results import Algo, Result


def _rademacher_projection_matrix(d: int, r: int, rng: np.random.Generator) -> np.ndarray:
    """
    Algorithm 2 in Boutsidis et al. (2014): R_{ij} in {+1/sqrt(r), -1/sqrt(r)} w.p. 1/2 each.
    Returns a dense float32 matrix (d x r).
    """
    signs = rng.integers(0, 2, size=(d, r), dtype=np.int8)
    signs = (signs * 2 - 1).astype(np.float32)
    signs *= (1.0 / math.sqrt(r))
    return signs


@dataclass
class StreamingState:
    sum_x: np.ndarray  # (k, d)
    count: np.ndarray  # (k,)


class Boutsidis_Streaming(Algo):
    name = "[Boutsidis2014] RandomProj Streaming"

    def __init__(
        self,
        eps: float = 0.3,
        c2: float = 8.0,
        r_min: int = 10,
        chunk_size: int = 1024,
    ):
        self.eps = float(eps)
        self.c2 = float(c2)
        self.r_min = int(r_min)
        self.chunk_size = int(chunk_size)

        self.R_: Optional[np.ndarray] = None
        self.km_: Optional[MiniBatchKMeans] = None
        self.state_: Optional[StreamingState] = None

    def _choose_r(self, d: int, k: int) -> int:
        r = int(math.ceil(self.c2 * k / (self.eps ** 2)))
        r = max(self.r_min, r)
        return min(d, r)

    def _init_models(
        self, d: int, k: int, rng: np.random.Generator
    ) -> Tuple[int, np.ndarray, MiniBatchKMeans, StreamingState]:
        r = self._choose_r(d, k)
        R = _rademacher_projection_matrix(d, r, rng)
        km = MiniBatchKMeans(
            n_clusters=k,
            batch_size=self.chunk_size,
            random_state=int(rng.integers(1, 1_000_000)),
        )
        state = StreamingState(
            sum_x=np.zeros((k, d), dtype=np.float64),
            count=np.zeros((k,), dtype=np.int64),
        )
        return r, R, km, state

    @staticmethod
    def _is_single_pass_iterable(obj: Any) -> bool:
        """
        Heuristic: generators/iterators usually return themselves from iter(obj).
        Re-iterable containers (list, tuple, etc.) return a NEW iterator each time.
        """
        try:
            it1 = iter(obj)
            it2 = iter(obj)
            return it1 is it2
        except TypeError:
            return True

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

    def _cost_against_centers(self, X: np.ndarray, centers: np.ndarray) -> float:
        X_sq_norms = self._squared_norms(X)
        dist2 = self._squared_distances(X, X_sq_norms, centers)
        return float(np.sum(np.min(dist2, axis=1)))

    def _centers_from_state(self, d: int, k: int) -> np.ndarray:
        assert self.state_ is not None
        centers = np.zeros((k, d), dtype=np.float64)
        counts = self.state_.count.astype(np.float64, copy=False)
        valid = counts > 0
        if np.any(valid):
            centers[valid] = self.state_.sum_x[valid] / counts[valid, None]
        return centers

    def _update_state(self, Xb: np.ndarray, pred: np.ndarray, k: int) -> None:
        assert self.state_ is not None
        batch_count = np.bincount(pred, minlength=k).astype(np.int64, copy=False)
        self.state_.count += batch_count
        np.add.at(self.state_.sum_x, pred, Xb)

    def _train_batches(
        self,
        batches: Iterable[np.ndarray],
        k: int,
        rng: np.random.Generator,
        labels_batches: Optional[Iterable[np.ndarray]] = None,
        compute_online_sse: bool = False,
    ) -> tuple[int, int, float, list[float], list[np.ndarray], list[np.ndarray]]:
        labels_iter = iter(labels_batches) if labels_batches is not None else None

        total_points = 0
        sse_online = 0.0
        all_true: list[np.ndarray] = []
        all_pred: list[np.ndarray] = []
        batch_times: list[float] = []

        initialized = False
        d: Optional[int] = None
        r: Optional[int] = None

        for Xb in batches:
            if Xb is None or len(Xb) == 0:
                continue

            Xb = np.asarray(Xb)
            if Xb.ndim != 2:
                raise ValueError(f"Each batch must be 2D (n_batch, d). Got shape={Xb.shape}")

            if not initialized:
                d = int(Xb.shape[1])
                r, R, km, state = self._init_models(d=d, k=k, rng=rng)
                self.R_, self.km_, self.state_ = R, km, state
                initialized = True

            tb0 = time.perf_counter()

            Xr = Xb @ self.R_
            self.km_.partial_fit(Xr)
            pred = self.km_.predict(Xr)
            self._update_state(Xb, pred, k)

            if compute_online_sse:
                centers_now = self._centers_from_state(d=d, k=k)
                sse_online += self._cost_against_centers(Xb, centers_now)

            total_points += Xb.shape[0]

            if labels_iter is not None:
                yb = next(labels_iter)
                all_true.append(np.asarray(yb))
                all_pred.append(pred.copy())

            tb1 = time.perf_counter()
            batch_times.append(tb1 - tb0)

        if not initialized or d is None or r is None:
            raise ValueError("No non-empty batches were provided.")

        return d, r, sse_online, batch_times, all_true, all_pred

    def fit_batches(
        self,
        batches: Iterable[np.ndarray],
        k: int,
        rng: np.random.Generator,
        labels_batches: Optional[Iterable[np.ndarray]] = None,
    ) -> Result:
        """
        True streaming API: consumes an iterable of batches.

        SSE policy:
          - If batches is re-iterable => do a second pass and compute exact SSE vs final centers.
          - If batches is single-pass => compute an online approximation against current lifted centers.
        """
        t0 = time.perf_counter()

        single_pass = self._is_single_pass_iterable(batches)
        d, r, sse_online, batch_times, all_true, all_pred = self._train_batches(
            batches=batches,
            k=k,
            rng=rng,
            labels_batches=labels_batches,
            compute_online_sse=single_pass,
        )

        centers_orig = self._centers_from_state(d=d, k=k)

        if single_pass:
            cost_sse = float(sse_online)
            cost_is_approx = True
        else:
            cost_sse = 0.0
            for Xb in batches:
                if Xb is None or len(Xb) == 0:
                    continue
                Xb = np.asarray(Xb)
                if Xb.ndim != 2:
                    raise ValueError(f"Each batch must be 2D (n_batch, d). Got shape={Xb.shape}")
                cost_sse += self._cost_against_centers(Xb, centers_orig)
            cost_is_approx = False

        ari = None
        nmi = None
        if all_true:
            y_true = np.concatenate(all_true, axis=0)
            y_pred = np.concatenate(all_pred, axis=0)
            ari = adjusted_rand_score(y_true, y_pred)
            nmi = normalized_mutual_info_score(y_true, y_pred)

        state_bytes = 0
        if self.R_ is not None:
            state_bytes += int(self.R_.nbytes)
        if self.state_ is not None:
            state_bytes += int(self.state_.sum_x.nbytes + self.state_.count.nbytes)
        if self.km_ is not None and hasattr(self.km_, "cluster_centers_") and self.km_.cluster_centers_ is not None:
            state_bytes += int(self.km_.cluster_centers_.nbytes)

        avg_update_ms = float(np.mean(batch_times) * 1000.0) if batch_times else float("nan")
        total_points = int(np.sum(self.state_.count)) if self.state_ is not None else 0

        t1 = time.perf_counter()

        return Result(
            centers=centers_orig.astype(np.float32),
            runtime_sec=t1 - t0,
            memory=float(state_bytes),
            cost_sse=float(cost_sse),
            cost_ratio_vs_kmeans=float("nan"),
            ari=ari,
            nmi=nmi,
            extra={
                "mode": "streaming",
                "eps": float(self.eps),
                "c2": float(self.c2),
                "chunk_size": int(self.chunk_size),
                "d": int(d),
                "r": int(r),
                "points_seen": total_points,
                "cost_is_approx": bool(cost_is_approx),
                "stream_is_single_pass": bool(single_pass),
                "avg_update_ms": float(avg_update_ms),
            },
        )

    def fit(self, samples: np.ndarray, k: int, rng: np.random.Generator, labels=None) -> Result:
        samples = np.asarray(samples)
        if samples.ndim != 2:
            raise ValueError(f"Samples must be 2D. Got shape={samples.shape}")

        t0 = time.perf_counter()

        def _batch_iter() -> Iterator[np.ndarray]:
            n = samples.shape[0]
            bs = max(1, self.chunk_size)
            for i in range(0, n, bs):
                yield samples[i:i + bs]

        y_batches = None
        if labels is not None:
            labels = np.asarray(labels)

            def _y_batch_iter() -> Iterator[np.ndarray]:
                n = labels.shape[0]
                bs = max(1, self.chunk_size)
                for i in range(0, n, bs):
                    yield labels[i:i + bs]

            y_batches = _y_batch_iter()

        d, r, _, batch_times, all_true, all_pred = self._train_batches(
            batches=_batch_iter(),
            k=k,
            rng=rng,
            labels_batches=y_batches,
            compute_online_sse=False,
        )

        centers_orig = self._centers_from_state(d=d, k=k)
        cost_sse = self._cost_against_centers(samples, centers_orig)

        ari = None
        nmi = None
        if all_true:
            y_true = np.concatenate(all_true, axis=0)
            y_pred = np.concatenate(all_pred, axis=0)
            ari = adjusted_rand_score(y_true, y_pred)
            nmi = normalized_mutual_info_score(y_true, y_pred)

        state_bytes = 0
        if self.R_ is not None:
            state_bytes += int(self.R_.nbytes)
        if self.state_ is not None:
            state_bytes += int(self.state_.sum_x.nbytes + self.state_.count.nbytes)
        if self.km_ is not None and hasattr(self.km_, "cluster_centers_") and self.km_.cluster_centers_ is not None:
            state_bytes += int(self.km_.cluster_centers_.nbytes)

        avg_update_ms = float(np.mean(batch_times) * 1000.0) if batch_times else float("nan")

        t1 = time.perf_counter()

        return Result(
            centers=centers_orig.astype(np.float32),
            runtime_sec=t1 - t0,
            memory=float(state_bytes),
            cost_sse=float(cost_sse),
            cost_ratio_vs_kmeans=float("nan"),
            ari=ari,
            nmi=nmi,
            extra={
                "mode": "batch_stream_wrapper",
                "eps": float(self.eps),
                "c2": float(self.c2),
                "chunk_size": int(self.chunk_size),
                "d": int(d),
                "r": int(r),
                "points_seen": int(samples.shape[0]),
                "cost_is_approx": False,
                "stream_is_single_pass": False,
                "avg_update_ms": float(avg_update_ms),
            },
        )
