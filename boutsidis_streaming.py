from __future__ import annotations
import time
import math
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional, Dict, Any, Tuple

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from results import Algo, Result
from utils import kmeans_cost_sse


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
    name = "Boutsidis2014_RandomProj_STREAMING"

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

    def _centers_from_state(self, d: int, k: int) -> np.ndarray:
        assert self.state_ is not None
        centers = np.zeros((k, d), dtype=np.float64)
        for j in range(k):
            c = int(self.state_.count[j])
            if c > 0:
                centers[j] = self.state_.sum_x[j] / float(c)
            else:
                centers[j] = 0.0
        return centers

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

    def fit_batches(
        self,
        batches: Iterable[np.ndarray],
        k: int,
        rng: np.random.Generator,
        labels_batches: Optional[Iterable[np.ndarray]] = None,
    ) -> Result:
        """
        True streaming API: consumes an iterable of batches.

        SSE policy (no flag, always computed):
          - If batches is re-iterable => do a second pass and compute EXACT SSE vs final centers.
          - If batches is single-pass (e.g., generator) => compute an ON-THE-FLY APPROX SSE
            (against current lifted centers during training).
        """
        t0 = time.perf_counter()

        single_pass = self._is_single_pass_iterable(batches)
        labels_iter = iter(labels_batches) if labels_batches is not None else None

        total_points = 0
        sse_online_approx = 0.0
        all_true = []
        all_pred = []

        initialized = False
        d: Optional[int] = None
        r: Optional[int] = None

        batch_times = []

    # ---- First pass: train streaming kmeans and build lifting state ----
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

            tb0 = time.perf_counter()  # ✅ start timing THIS batch update

            # Project this batch: C_t = X_t R
            Xr = Xb @ self.R_  # (b, r)

            # Update k-means in reduced space
            self.km_.partial_fit(Xr)

            # Assign labels using current model
            pred = self.km_.predict(Xr)

            # Update original-space sufficient statistics (sum/count)
            for j in range(k):
                mask = (pred == j)
                if np.any(mask):
                    self.state_.sum_x[j] += Xb[mask].sum(axis=0)
                    self.state_.count[j] += int(mask.sum())

            # Always compute SOME SSE:
            if single_pass:
                centers_now = self._centers_from_state(d=d, k=k)
                diff = Xb[:, None, :] - centers_now[None, :, :]
                dist2 = np.sum(diff * diff, axis=2)
                sse_online_approx += float(np.sum(np.min(dist2, axis=1)))

            total_points += Xb.shape[0]

            if labels_iter is not None:
                yb = next(labels_iter)
                all_true.append(np.asarray(yb))
                all_pred.append(pred.copy())

            tb1 = time.perf_counter()   # ✅ end timing THIS batch update
            batch_times.append(tb1 - tb0)

        if not initialized:
            raise ValueError("No non-empty batches were provided.")

        # Final centers in original space
        centers_orig = self._centers_from_state(d=d, k=k)
        
        # ---- SSE computation ----
        if single_pass:
            # Can't do exact SSE without seeing the stream again
            cost_sse = float(sse_online_approx)
            cost_is_approx = True
        else:
            # Second pass: exact SSE against final centers
            cost_sse = 0.0
            for Xb in batches:  # re-iterate
                if Xb is None or len(Xb) == 0:
                    continue
                Xb = np.asarray(Xb)
                if Xb.ndim != 2:
                    raise ValueError(f"Each batch must be 2D (n_batch, d). Got shape={Xb.shape}")
                diff = Xb[:, None, :] - centers_orig[None, :, :]
                dist2 = np.sum(diff * diff, axis=2)
                cost_sse += float(np.sum(np.min(dist2, axis=1)))
            cost_is_approx = False

        # Metrics: only if streamed labels provided (ARI/NMI computed for predictions collected in pass 1)
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
                "mode": "streaming",
                "eps": float(self.eps),
                "c2": float(self.c2),
                "chunk_size": int(self.chunk_size),
                "d": int(d),
                "r": int(r),
                "points_seen": int(total_points),
                "cost_is_approx": bool(cost_is_approx),
                "stream_is_single_pass": bool(single_pass),
                "avg_update_ms": float(avg_update_ms),
                "state_bytes": int(state_bytes),
                "memory_mb": float(state_bytes / (1024.0 ** 2)),
            },
        )

    def fit(self, samples: np.ndarray, k: int, rng: np.random.Generator, labels=None) -> Result:
        samples = np.asarray(samples)
        if samples.ndim != 2:
            raise ValueError(f"Samples must be 2D. Got shape={samples.shape}")

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

        # Here batches are generated from an in-memory array, so we can just compute exact SSE directly
        # after training (more reliable than relying on re-iterability heuristics).
        res = self.fit_batches(_batch_iter(), k=k, rng=rng, labels_batches=y_batches)

        # Override SSE with exact one-pass over full samples vs final centers (always exact here)
        cost = kmeans_cost_sse(samples, res.centers.astype(np.float64))
        res.cost_sse = float(cost)
        if res.extra is None:
            res.extra = {}
        res.extra["mode"] = "batch_stream_wrapper"
        res.extra["cost_is_approx"] = False
        res.extra["stream_is_single_pass"] = False
        return res
