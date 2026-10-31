from __future__ import annotations

import numpy as np

from .detectors import robust_location_scale


class NormalTransitionModel:
    """Ridge one-step predictor with robustly scaled transition residual scores.

    This is a normal-data statistical surrogate, not an identified plant model or
    safety certificate. It uses paired current/first-difference observations so
    training rows need not be contiguous after deterministic sampling.
    """

    def __init__(self, ridge: float = 1.0, top_k: int = 5):
        if ridge < 0:
            raise ValueError("ridge must be nonnegative")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self.ridge = float(ridge)
        self.top_k = int(top_k)

    def fit(self, x: np.ndarray, dx: np.ndarray) -> "NormalTransitionModel":
        x = np.asarray(x, dtype=float)
        dx = np.asarray(dx, dtype=float)
        if x.ndim != 2 or dx.shape != x.shape or len(x) < 2:
            raise ValueError("x and dx must be matching 2-D arrays with at least two rows")
        if not np.isfinite(x).all() or not np.isfinite(dx).all():
            raise ValueError("transition training data must be finite")
        self.center, self.level_scale = robust_location_scale(x)
        previous = x - dx
        z = (x - self.center) / self.level_scale
        p = (previous - self.center) / self.level_scale
        p_mean = p.mean(axis=0)
        z_mean = z.mean(axis=0)
        pc = p - p_mean
        zc = z - z_mean
        gram = (pc.T @ pc) / len(x)
        cross = (pc.T @ zc) / len(x)
        gram.flat[::gram.shape[0] + 1] += self.ridge
        self.coef = np.linalg.solve(gram, cross)
        self.bias = z_mean - p_mean @ self.coef
        residual = z - (p @ self.coef + self.bias)
        self.residual_center, self.residual_scale = robust_location_scale(residual)
        train_score = self.score(previous, x)
        self.fit_metadata = {
            "rows": int(len(x)), "features": int(x.shape[1]), "ridge": self.ridge,
            "top_k_residuals": min(self.top_k, int(x.shape[1])),
            "training_score_q95": float(np.quantile(train_score, 0.95)),
            "scope": "normal-only ridge one-step statistical model; not a validated plant model",
        }
        return self

    def score(self, previous_x: np.ndarray, current_x: np.ndarray) -> np.ndarray:
        previous_x = np.asarray(previous_x, dtype=float)
        current_x = np.asarray(current_x, dtype=float)
        one = previous_x.ndim == 1
        if one:
            previous_x = previous_x[None, :]
            current_x = current_x[None, :]
        if previous_x.ndim != 2 or current_x.shape != previous_x.shape:
            raise ValueError("previous_x and current_x must be matching 1-D or 2-D arrays")
        if previous_x.shape[1] != len(self.center):
            raise ValueError("transition sample feature count differs from fitted schema")
        zp = (previous_x - self.center) / self.level_scale
        zc = (current_x - self.center) / self.level_scale
        residual = zc - (zp @ self.coef + self.bias)
        standardized = np.abs((residual - self.residual_center) / self.residual_scale)
        k = min(self.top_k, standardized.shape[1])
        return np.partition(standardized, -k, axis=1)[:, -k:].mean(axis=1)

    def calibrate(self, previous_x: np.ndarray, current_x: np.ndarray,
                  false_positive_rate: float = 0.05) -> float:
        if not 0.0 < false_positive_rate < 1.0:
            raise ValueError("false_positive_rate must be strictly between zero and one")
        scores = self.score(previous_x, current_x)
        return float(np.quantile(scores, 1.0 - false_positive_rate))
