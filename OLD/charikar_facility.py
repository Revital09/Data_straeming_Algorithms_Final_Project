from __future__ import annotations
import time
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from results import Algo, Result
from utils import weighted_kmeans_centers, assign_labels, kmeans_cost_sse


@dataclass
class _SummaryPoint:
    x: np.ndarray
    w: float


class _OnlineFLKMeansStateFast:
    """
    Faster ONLINE-FL-style state.

    Key optimizations:
    - centers stored in one ndarray, not list of arrays
    - counts stored in one ndarray
    - no np.vstack on every point
    - rollback only undoes the last operation instead of copying whole state
    """

    def __init__(self, facility_cost: float, d: int):
        self.facility_cost = float(facility_cost)
        self.d = int(d)

        self.centers: Optional[np.ndarray] = None   # shape (m, d)
        self.counts: Optional[np.ndarray] = None    # shape (m,)

        self.total_cost: float = 0.0
        self.num_opened: int = 0
        self.processed_items: int = 0
        self.raw_points_read: int = 0

        self.stopped: bool = False
        self.stop_reason: str | None = None

        # rollback info for only the most recent processed point
        self._last_action: Optional[str] = None
        self._last_index: int = -1
        self._last_prev_count: float = 0.0
        self._last_added_cost: float = 0.0
        self._last_w: float = 0.0
        self._last_was_raw: bool = False

    @property
    def num_centers(self) -> int:
        return 0 if self.centers is None else int(self.centers.shape[0])

    def _nearest_center_sqdist(self, x: np.ndarray) -> Tuple[int, float]:
        # centers shape: (m, d)
        diff = self.centers - x
        sq_dists = np.einsum("ij,ij->i", diff, diff, optimize=True)
        j = int(np.argmin(sq_dists))
        return j, float(sq_dists[j])

    def process_point(self, x: np.ndarray, w: float, is_raw: bool, rng: np.random.Generator) -> None:
        self._last_action = None
        self._last_index = -1
        self._last_prev_count = 0.0
        self._last_added_cost = 0.0
        self._last_w = float(w)
        self._last_was_raw = bool(is_raw)

        if self.centers is None:
            self.centers = x.reshape(1, self.d).astype(np.float64, copy=True)
            self.counts = np.array([w], dtype=np.float64)
            self.num_opened = 1
            self.processed_items = 1
            if is_raw:
                self.raw_points_read = 1
            self._last_action = "open_first"
            return

        j, d2 = self._nearest_center_sqdist(x)
        p_open = min(1.0, (w * d2) / (self.facility_cost + 1e-12))

        if rng.random() < p_open:
            self.centers = np.vstack([self.centers, x.reshape(1, self.d)])
            self.counts = np.append(self.counts, w)
            self.num_opened += 1
            self._last_action = "open_new"
            self._last_index = self.num_centers - 1
        else:
            self._last_action = "assign"
            self._last_index = j
            self._last_prev_count = float(self.counts[j])
            self._last_added_cost = w * d2
            self.counts[j] += w
            self.total_cost += self._last_added_cost

        self.processed_items += 1
        if is_raw:
            self.raw_points_read += 1

    def rollback_last(self) -> None:
        """
        Undo the last processed point only.
        """
        if self._last_action is None:
            return

        if self._last_action == "open_first":
            self.centers = None
            self.counts = None
            self.num_opened = 0

        elif self._last_action == "open_new":
            self.centers = self.centers[:-1]
            self.counts = self.counts[:-1]
            self.num_opened -= 1

        elif self._last_action == "assign":
            self.counts[self._last_index] = self._last_prev_count
            self.total_cost -= self._last_added_cost

        self.processed_items -= 1
        if self._last_was_raw:
            self.raw_points_read -= 1

        self._last_action = None

    def snapshot(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.centers is None or self.counts is None or self.centers.shape[0] == 0:
            raise RuntimeError("Invocation has no centers.")
        return self.centers.copy(), self.counts.copy()


class CharikarInspired_PLS_KMeans_ChunkLoop(Algo):
    """
    Charikar-inspired PLS skeleton adapted to k-means, optimized.

    Main speedups:
    - no full state copies for rollback
    - no per-point WeightedPoint allocation for raw data
    - no np.vstack inside nearest-center computation
    """

    name = "[16]CharikarInspired_PLS_KMeans_ChunkLoop"

    def __init__(
        self,
        beta: float = 25.0,
        gamma: float = 100.0,
        chunk_size: int = 1000,
        n_init_final: int = 5,
        max_iter_final: int = 300,
    ):
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.chunk_size = int(chunk_size)
        self.n_init_final = int(n_init_final)
        self.max_iter_final = int(max_iter_final)

    def _set_lb_kmeans(self, X: np.ndarray, k: int) -> float:
        m = min(X.shape[0], k + 1)
        if m <= 1:
            return 1.0

        Y = X[:m].astype(np.float64, copy=False)
        diff = Y[:, None, :] - Y[None, :, :]
        d2 = np.einsum("ijk,ijk->ij", diff, diff, optimize=True)
        np.fill_diagonal(d2, np.inf)
        lb = float(np.min(d2))
        return max(lb, 1e-12)

    def _init_phase_states(
        self,
        Li: float,
        k: int,
        n: int,
        d: int,
        rng: np.random.Generator,
    ):
        logn = max(1.0, math.log(max(2, n)))
        num_runs = max(1, int(math.ceil(2.0 * logn)))  # paper-style 2 log n runs
        facility_cost = Li / (k * (1.0 + logn) + 1e-12)

        median_limit = int(math.ceil(
            4.0 * k * (1.0 + logn) * (1.0 + 4.0 * (self.gamma + self.beta))
        ))
        cost_limit = 4.0 * Li * (1.0 + 4.0 * (self.gamma + self.beta))

        states = [_OnlineFLKMeansStateFast(facility_cost=facility_cost, d=d) for _ in range(num_runs)]
        run_seeds = rng.integers(0, 2**32 - 1, size=num_runs, dtype=np.uint64)
        run_rngs = [np.random.default_rng(int(s)) for s in run_seeds]

        return states, run_rngs, facility_cost, median_limit, cost_limit, num_runs

    def _feed_summary_to_states(
        self,
        states: list[_OnlineFLKMeansStateFast],
        run_rngs: list[np.random.Generator],
        summary_points: list[_SummaryPoint],
        median_limit: int,
        cost_limit: float,
    ) -> None:
        for p in summary_points:
            any_active = False
            for st, rrng in zip(states, run_rngs):
                if st.stopped:
                    continue
                any_active = True

                st.process_point(p.x, p.w, False, rrng)

                overflow = (st.num_opened > median_limit) or (st.total_cost > cost_limit)
                if overflow:
                    st.rollback_last()
                    st.stopped = True
                    st.stop_reason = "threshold_exceeded"

            if not any_active:
                break

    def _feed_raw_chunk_to_states(
        self,
        states: list[_OnlineFLKMeansStateFast],
        run_rngs: list[np.random.Generator],
        chunk: np.ndarray,
        median_limit: int,
        cost_limit: float,
    ) -> None:
        for x in chunk:
            any_active = False
            for st, rrng in zip(states, run_rngs):
                if st.stopped:
                    continue
                any_active = True

                st.process_point(x, 1.0, True, rrng)

                overflow = (st.num_opened > median_limit) or (st.total_cost > cost_limit)
                if overflow:
                    st.rollback_last()
                    st.stopped = True
                    st.stop_reason = "threshold_exceeded"

            if not any_active:
                break

    def _run_one_phase_chunked(
        self,
        X: np.ndarray,
        raw_start_idx: int,
        summary_points: list[_SummaryPoint],
        Li: float,
        k: int,
        rng: np.random.Generator,
    ) -> tuple[list[_SummaryPoint], int, dict]:
        n, d = X.shape

        states, run_rngs, facility_cost, median_limit, cost_limit, num_runs = self._init_phase_states(
            Li=Li, k=k, n=n, d=d, rng=rng
        )

        if summary_points:
            self._feed_summary_to_states(
                states=states,
                run_rngs=run_rngs,
                summary_points=summary_points,
                median_limit=median_limit,
                cost_limit=cost_limit,
            )

        for start in range(raw_start_idx, n, self.chunk_size):
            stop = min(start + self.chunk_size, n)
            self._feed_raw_chunk_to_states(
                states=states,
                run_rngs=run_rngs,
                chunk=X[start:stop],
                median_limit=median_limit,
                cost_limit=cost_limit,
            )
            if all(st.stopped for st in states):
                break

        for st in states:
            if not st.stopped:
                st.stop_reason = "end_of_stream"

        winner = max(states, key=lambda s: s.processed_items)
        Cw, Ww = winner.snapshot()

        Mi = [_SummaryPoint(x=Cw[i].copy(), w=float(Ww[i])) for i in range(Cw.shape[0])]

        stats = {
            "facility_cost": float(facility_cost),
            "median_limit": int(median_limit),
            "cost_limit": float(cost_limit),
            "winner_processed_items": int(winner.processed_items),
            "winner_raw_points_read": int(winner.raw_points_read),
            "winner_num_centers": int(Cw.shape[0]),
            "winner_cost": float(winner.total_cost),
            "num_parallel_runs": int(num_runs),
        }

        return Mi, int(winner.raw_points_read), stats

    def fit(self, X: np.ndarray, k: int, rng: np.random.Generator, y=None) -> Result:
        t0 = time.perf_counter()

        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape

        if n == 0:
            raise ValueError("X must contain at least one sample.")
        if k <= 0:
            raise ValueError("k must be positive.")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")

        L = self._set_lb_kmeans(X, k) / self.beta

        raw_start_idx = 0
        phase_id = 1
        summary_points: list[_SummaryPoint] = []
        phase_summaries: list[dict] = []

        while raw_start_idx < n:
            Mi, raw_consumed, phase_stats = self._run_one_phase_chunked(
                X=X,
                raw_start_idx=raw_start_idx,
                summary_points=summary_points,
                Li=L,
                k=k,
                rng=rng,
            )

            phase_summaries.append(
                {
                    "phase": int(phase_id),
                    "Li": float(L),
                    "raw_start_index": int(raw_start_idx),
                    "raw_consumed": int(raw_consumed),
                    "summary_in_size": int(len(summary_points)),
                    "summary_out_size": int(len(Mi)),
                    "chunk_size": int(self.chunk_size),
                    **phase_stats,
                }
            )

            summary_points = Mi

            if raw_consumed <= 0:
                summary_points.append(_SummaryPoint(x=X[raw_start_idx].copy(), w=1.0))
                raw_start_idx += 1
            else:
                raw_start_idx += raw_consumed

            L *= self.beta
            phase_id += 1

        Csum = np.vstack([p.x for p in summary_points]).astype(np.float64, copy=False)
        Wsum = np.asarray([p.w for p in summary_points], dtype=np.float64)

        if Csum.shape[0] > k:
            centers_final = weighted_kmeans_centers(
                Csum,
                Wsum,
                k=k,
                rng=rng,
                n_init=self.n_init_final,
                max_iter=self.max_iter_final,
            )
        elif Csum.shape[0] == k:
            centers_final = Csum.copy()
        else:
            extra_idx = rng.integers(0, Csum.shape[0], size=k - Csum.shape[0])
            centers_final = np.vstack([Csum, Csum[extra_idx]])

        t1 = time.perf_counter()

        cost = float(kmeans_cost_sse(X, centers_final))
        pred = assign_labels(X, centers_final)

        ari = adjusted_rand_score(y, pred) if y is not None else None
        nmi = normalized_mutual_info_score(y, pred) if y is not None else None

        state_bytes = int(Csum.nbytes + Wsum.nbytes)

        return Result(
            centers=centers_final,
            runtime_sec=float(t1 - t0),
            memory=state_bytes,
            cost_sse=cost,
            cost_ratio_vs_kmeans=float("nan"),
            ari=ari,
            nmi=nmi,
            extra={
                "points_seen": int(n),
                "dimension": int(d),
                "num_phases": int(phase_id - 1),
                "final_summary_size": int(Csum.shape[0]),
                "final_lower_bound": float(L / self.beta),
                "avg_update_ms": float((t1 - t0) * 1000.0 / max(1, n)),
                "chunk_size": int(self.chunk_size),
                "phase_summaries": phase_summaries,
                "beta": float(self.beta),
                "gamma": float(self.gamma),
            },
        )