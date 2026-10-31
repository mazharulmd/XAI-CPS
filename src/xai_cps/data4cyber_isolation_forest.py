from __future__ import annotations

import numpy as np

from xai_cps.detectors import robust_location_scale, valid_references


class Data4CyberIsolationForest:
    """Normal-only Isolation Forest on robust-scaled process levels and deltas.

    The forest is a detector-family reference; it has no native per-channel
    attribution. Explanation rankings must therefore use a separately reported
    black-box score ablation, with levels and temporal differences kept aligned.
    """

    name = "IsolationForest-TEMPORAL"

    def __init__(self, random_state: int = 42, n_estimators: int = 500,
                 max_samples: int = 512) -> None:
        if n_estimators < 1 or max_samples < 2:
            raise ValueError("n_estimators and max_samples must be positive")
        self.random_state = int(random_state)
        self.n_estimators = int(n_estimators)
        self.max_samples = int(max_samples)

    def _join(self, x: np.ndarray, dx: np.ndarray) -> np.ndarray:
        zx = np.clip((np.asarray(x, dtype=float) - self.x_med) / self.x_scale, -1e3, 1e3)
        zd = np.clip((np.asarray(dx, dtype=float) - self.dx_med) / self.dx_scale, -1e3, 1e3)
        if zx.ndim != 2 or zd.shape != zx.shape:
            raise ValueError("x and dx must be matching two-dimensional arrays")
        return np.concatenate((zx, zd), axis=1)

    def fit(self, x: np.ndarray, dx: np.ndarray) -> "Data4CyberIsolationForest":
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError("Install the optional baselines extra to use Isolation Forest") from error
        x = np.asarray(x, dtype=float)
        dx = np.asarray(dx, dtype=float)
        if x.ndim != 2 or dx.shape != x.shape or not np.isfinite(x).all() or not np.isfinite(dx).all():
            raise ValueError("training x and dx must be matching finite matrices")
        self.x_med, self.x_scale = robust_location_scale(x)
        self.dx_med, self.dx_scale = robust_location_scale(dx)
        self.x_ref = valid_references(x)
        self.dx_ref = self.dx_med.copy()
        z = self._join(x, dx)
        self.forest_ = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=min(self.max_samples, len(z)),
            contamination="auto",
            random_state=self.random_state,
            n_jobs=1,
        ).fit(z)
        self.n_features_in_ = x.shape[1]
        return self

    def score(self, x: np.ndarray, dx: np.ndarray) -> np.ndarray:
        if not hasattr(self, "forest_"):
            raise RuntimeError("fit must be called before score")
        z = self._join(x, dx)
        if z.shape[1] != 2 * self.n_features_in_:
            raise ValueError("input feature count differs from the fitted schema")
        # sklearn's score_samples is lower for outliers; use a higher-is-more-
        # anomalous convention to match the rest of this repository.
        return -np.asarray(self.forest_.score_samples(z), dtype=float)

