from __future__ import annotations
import time
import math

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from results import Algo, Result
from utils import weighted_kmeans_centers, assign_labels, kmeans_cost_sse


class _OnlineFLKMeansState:
    """
    One ONLINE-FL-style run adapted to k-means.

    Optimizations:
    - preallocated center/count buffers
    - cached center squared norms
    - rollback only the last local update
    - no per-point vstack/list copying
    """

    def __init__(self, facility_cost: float, d: int, max_centers: int):
        self.facility_cost = float(facility_cost)
        self.d = int(d)
        self.max_centers = int(max_centers)

        self.centers = np.zeros((max_centers, d), dtype=np.float64)
        self.center_sq_norms = np.zeros(max_centers, dtype=np.float64)
        self.counts = np.zeros(max_centers, dtype=np.float64)

        self.total_cost: float = 0.0
        self.num_opened: int = 0
        self.processed_items: int = 0
        self.raw_points_read: int = 0

        self.stopped: bool = False
        self.stop_reason: str | None = None

        # rollback fields for the most recent processed point
        self._undo_prev_total_cost: float = 0.0
        self._undo_prev_processed: int = 0
        self._undo_prev_raw_read: int = 0
        self._undo_action: int = 0   # 0=none, 1=open, 2=assign
        self._undo_index: int = -1
        self._undo_prev_count: float = 0.0

    def process_point(
        self,
        x: np.ndarray,
        w: float,
        is_raw: bool,
        rng: np.random.Generator,
    ) -> None:
        """
        Process one weighted point.
        Save only enough information to undo the last local update.
        """
        w = float(w)

        self._undo_prev_total_cost = self.total_cost
        self._undo_prev_processed = self.processed_items
        self._undo_prev_raw_read = self.raw_points_read
        self._undo_action = 0
        self._undo_index = -1
        self._undo_prev_count = 0.0

        if self.num_opened == 0:
            idx = 0
            self.centers[idx] = x
            self.center_sq_norms[idx] = float(np.dot(x, x))
            self.counts[idx] = w
            self.num_opened = 1

            self._undo_action = 1
            self._undo_index = idx
        else:
            active_centers = self.centers[:self.num_opened]
            x_norm2 = float(np.dot(x, x))

            # ||c-x||^2 = ||c||^2 + ||x||^2 - 2<c,x>
            sq_dists = (
                self.center_sq_norms[:self.num_opened]
                + x_norm2
                - 2.0 * (active_centers @ x)
            )
            sq_dists = np.maximum(sq_dists, 0.0)

            j = int(np.argmin(sq_dists))
            d2 = float(sq_dists[j])

            # k-means adaptation of Meyerson/Charikar opening rule
            p_open = min(1.0, (w * d2) / (self.facility_cost + 1e-12))

            if rng.random() < p_open:
                idx = self.num_opened
                self.centers[idx] = x
                self.center_sq_norms[idx] = x_norm2
                self.counts[idx] = w
                self.num_opened += 1

                self._undo_action = 1
                self._undo_index = idx
            else:
                self._undo_action = 2
                self._undo_index = j
                self._undo_prev_count = float(self.counts[j])

                self.counts[j] += w
                self.total_cost += w * d2

        self.processed_items += 1
        if is_raw:
            self.raw_points_read += 1

    def rollback_last(self) -> None:
        """
        Undo only the most recent processed point.
        """
        self.total_cost = self._undo_prev_total_cost
        self.processed_items = self._undo_prev_processed
        self.raw_points_read = self._undo_prev_raw_read

        if self._undo_action == 1:  # opened a center
            idx = self._undo_index
            self.counts[idx] = 0.0
            self.center_sq_norms[idx] = 0.0
            self.num_opened -= 1
        elif self._undo_action == 2:  # assigned to existing center
            idx = self._undo_index
            self.counts[idx] = self._undo_prev_count

        self._undo_action = 0
        self._undo_index = -1
        self._undo_prev_count = 0.0

    def snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        if self.num_opened == 0:
            raise RuntimeError("Invocation has no centers.")
        C = self.centers[:self.num_opened].copy()
        w = self.counts[:self.num_opened].copy()
        return C, w


class Charikar_KMeans(Algo):
    """
    Charikar-inspired PLS skeleton adapted to k-means.

    Paper-like structure:
    - SET-LB-style lower bound initialization
    - repeated phases
    - ~2 log n parallel ONLINE-FL runs per phase
    - next phase gets Mi || unread suffix
    - no extra summary reduction inside phases

    Practical adaptations:
    - squared-distance opening rule for k-means
    - final weighted k-means on the phase summary
    - raw points are processed in chunk loops inside fit()
    """

    name = "[16]Charikar_PLS_KMeans"

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
        """
        PLS-style lower bound initialization adapted to k-means:
        minimum squared pairwise distance among the first k+1 points.
        """
        m = min(X.shape[0], k + 1)
        if m <= 1:
            return 1.0

        Y = X[:m]
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

        # Paper-style: 2 log n parallel runs
        num_runs = max(1, int(math.ceil(2.0 * logn)))

        facility_cost = Li / (k * (1.0 + logn) + 1e-12)

        median_limit = int(math.ceil(
            4.0 * k * (1.0 + logn) * (1.0 + 4.0 * (self.gamma + self.beta))
        ))
        cost_limit = 4.0 * Li * (1.0 + 4.0 * (self.gamma + self.beta))

        # allow one extra temporary opening before rollback
        max_centers = max(1, median_limit + 1)

        states = [
            _OnlineFLKMeansState(
                facility_cost=facility_cost,
                d=d,
                max_centers=max_centers,
            )
            for _ in range(num_runs)
        ]

        run_seeds = rng.integers(0, 2**32 - 1, size=num_runs, dtype=np.uint64)
        run_rngs = [np.random.default_rng(int(s)) for s in run_seeds]

        return states, run_rngs, facility_cost, median_limit, cost_limit, num_runs

    def _feed_summary_to_states(
        self,
        states: list[_OnlineFLKMeansState],
        run_rngs: list[np.random.Generator],
        summary_X: np.ndarray,
        summary_w: np.ndarray,
        median_limit: int,
        cost_limit: float,
    ) -> None:
        m = summary_X.shape[0]
        for i in range(m):
            any_active = False
            x = summary_X[i]
            w = float(summary_w[i])

            for st, rrng in zip(states, run_rngs):
                if st.stopped:
                    continue
                any_active = True

                st.process_point(x=x, w=w, is_raw=False, rng=rrng)

                overflow = (st.num_opened > median_limit) or (st.total_cost > cost_limit)
                if overflow:
                    st.rollback_last()
                    st.stopped = True
                    st.stop_reason = "threshold_exceeded"

            if not any_active:
                break

    def _feed_raw_chunk_to_states(
        self,
        states: list[_OnlineFLKMeansState],
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

                st.process_point(x=x, w=1.0, is_raw=True, rng=rrng)

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
        summary_X: np.ndarray | None,
        summary_w: np.ndarray | None,
        Li: float,
        k: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, int, dict]:
        n, d = X.shape

        states, run_rngs, facility_cost, median_limit, cost_limit, num_runs = self._init_phase_states(
            Li=Li, k=k, n=n, d=d, rng=rng
        )

        # Feed carried summary Mi first
        if summary_X is not None and summary_w is not None and summary_X.shape[0] > 0:
            self._feed_summary_to_states(
                states=states,
                run_rngs=run_rngs,
                summary_X=summary_X,
                summary_w=summary_w,
                median_limit=median_limit,
                cost_limit=cost_limit,
            )

        # Then unread raw stream in chunks
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

        # PLS-style winner: run that progressed the farthest
        winner = max(states, key=lambda s: s.processed_items)
        Mi_X, Mi_w = winner.snapshot()

        stats = {
            "facility_cost": float(facility_cost),
            "median_limit": int(median_limit),
            "cost_limit": float(cost_limit),
            "winner_processed_items": int(winner.processed_items),
            "winner_raw_points_read": int(winner.raw_points_read),
            "winner_num_centers": int(Mi_X.shape[0]),
            "winner_cost": float(winner.total_cost),
            "num_parallel_runs": int(num_runs),
        }
        return Mi_X, Mi_w, int(winner.raw_points_read), stats

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

        # PLS-style initialization
        L = self._set_lb_kmeans(X, k) / self.beta

        raw_start_idx = 0
        phase_id = 1
        summary_X: np.ndarray | None = None
        summary_w: np.ndarray | None = None
        phase_summaries: list[dict] = []

        while raw_start_idx < n:
            summary_in_size = 0 if summary_X is None else int(summary_X.shape[0])

            Mi_X, Mi_w, raw_consumed, phase_stats = self._run_one_phase_chunked(
                X=X,
                raw_start_idx=raw_start_idx,
                summary_X=summary_X,
                summary_w=summary_w,
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
                    "summary_in_size": int(summary_in_size),
                    "summary_out_size": int(Mi_X.shape[0]),
                    "chunk_size": int(self.chunk_size),
                    **phase_stats,
                }
            )

            # Xi+1 = Mi || unread suffix
            summary_X = Mi_X
            summary_w = Mi_w

            if raw_consumed <= 0:
                # safety against deadlock
                x = X[raw_start_idx:raw_start_idx + 1]
                w = np.array([1.0], dtype=np.float64)

                if summary_X is None:
                    summary_X = x.copy()
                    summary_w = w
                else:
                    summary_X = np.vstack([summary_X, x])
                    summary_w = np.hstack([summary_w, w])

                raw_start_idx += 1
            else:
                raw_start_idx += raw_consumed

            L *= self.beta
            phase_id += 1

        assert summary_X is not None and summary_w is not None

        Csum = summary_X.astype(np.float64, copy=False)
        Wsum = summary_w.astype(np.float64, copy=False)

        # Final weighted k-means on the final PLS summary
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
            memory=float(state_bytes),
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