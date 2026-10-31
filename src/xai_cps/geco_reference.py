from __future__ import annotations

import numpy as np
import time


class GeCoAR1Reference:
    """Portable, constrained GeCo-style process detector for state streams.

    This implements the released GeCo residual/CUSUM recipe with the search
    restricted to each variable's own one-step lag (equation length zero in
    the upstream combination generator). It is an adapted reference, not the
    full cross-sensor GeCo implementation. All fitted quantities use normal
    training sequences only; alert calibration is performed by the caller.
    """

    name = "GeCo-AR1-constrained"

    def __init__(self, fit_fraction: float = 0.8) -> None:
        if not 0.0 < fit_fraction < 1.0:
            raise ValueError("fit_fraction must be between zero and one")
        self.fit_fraction = fit_fraction

    def fit(self, normal_sequences: list[np.ndarray]) -> "GeCoAR1Reference":
        pairs_x: list[np.ndarray] = []
        pairs_y: list[np.ndarray] = []
        n_features: int | None = None
        for sequence in normal_sequences:
            values = np.asarray(sequence, dtype=float)
            if values.ndim != 2 or len(values) < 3:
                continue
            if not np.isfinite(values).all():
                raise ValueError("GeCo-AR1 inputs must be finite after training-only imputation")
            if n_features is None:
                n_features = values.shape[1]
            elif values.shape[1] != n_features:
                raise ValueError("All normal sequences must use the same feature schema")
            pairs_x.append(values[:-1])
            pairs_y.append(values[1:])
        if not pairs_x or n_features is None:
            raise ValueError("At least one finite normal training sequence is required")

        previous = np.concatenate(pairs_x, axis=0)
        current = np.concatenate(pairs_y, axis=0)
        cut = int(self.fit_fraction * len(current))
        if cut < 2 or cut >= len(current):
            raise ValueError("Too few normal transitions for coefficient fitting and residual calibration")

        # Upstream GeCo's length-zero SUM equation is y_t = a*x_(t-1) + b.
        design = np.column_stack((previous[:cut], np.ones(cut)))
        coefficients = np.empty(n_features, dtype=float)
        intercepts = np.empty(n_features, dtype=float)
        for j in range(n_features):
            fit, *_ = np.linalg.lstsq(design[:, [j, -1]], current[:cut, j], rcond=None)
            coefficients[j], intercepts[j] = fit

        errors = coefficients[None, :] * previous + intercepts[None, :] - current
        self.drift_ = np.mean(np.abs(errors), axis=0) + np.std(errors, axis=0)
        self.coefficients_ = coefficients
        self.intercepts_ = intercepts
        self.training_cusum_limit_ = np.zeros(n_features, dtype=float)
        for row in errors:
            self.training_cusum_limit_ = np.maximum(
                self.training_cusum_limit_ + np.abs(row) - self.drift_, 0.0
            )
        self.training_cusum_limit_ = np.maximum(self.training_cusum_limit_, 1e-10)
        return self

    def score_sequence(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return per-channel normalized CUSUM and its row-wise maximum.

        CUSUM state resets at the start of each supplied scenario. The first
        row has score zero because no preceding state exists in this stream.
        """
        x = np.asarray(values, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(self.coefficients_):
            raise ValueError("Input must be a 2-D sequence with the fitted feature count")
        if not np.isfinite(x).all():
            raise ValueError("GeCo-AR1 inputs must be finite after training-only imputation")
        cusum = np.zeros(x.shape[1], dtype=float)
        channel_scores = np.zeros_like(x, dtype=float)
        latency_ms = np.zeros(len(x), dtype=float)
        for t in range(1, len(x)):
            started = time.perf_counter()
            error = self.coefficients_ * x[t - 1] + self.intercepts_ - x[t]
            cusum = np.maximum(cusum + np.abs(error) - self.drift_ - 1e-10, 0.0)
            channel_scores[t] = cusum / self.training_cusum_limit_
            latency_ms[t] = (time.perf_counter() - started) * 1000.0
        return channel_scores, channel_scores.max(axis=1, initial=0.0), latency_ms[1:]

