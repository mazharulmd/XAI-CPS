from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest

from .detectors import robust_location_scale, valid_references


class HAIIsolationForest:
    """Normal-only Isolation Forest over robust-scaled HAI state and deltas.

    This is a reproducible classical detector-family comparator, not a
    reproduction of RAID's regime clustering, physics-informed feature
    engineering, language generation, or RAG stages. Explanations are
    evaluated separately with the repository's identity-preserving channel
    deletion procedure.
    """

    name = "IsolationForest-TEMPORAL"

    def __init__(
        self,
        n_estimators: int = 500,
        max_samples: int = 512,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        if n_estimators < 1 or max_samples < 2:
            raise ValueError("n_estimators must be positive and max_samples at least two")
        self.n_estimators = int(n_estimators)
        self.max_samples = int(max_samples)
        self.random_state = int(random_state)
        self.n_jobs = int(n_jobs)

    def fit(self, x: np.ndarray, dx: np.ndarray) -> "HAIIsolationForest":
        x = np.asarray(x, dtype=float)
        dx = np.asarray(dx, dtype=float)
        if x.ndim != 2 or dx.shape != x.shape or len(x) < 2:
            raise ValueError("x and dx must be same-shaped 2-D arrays with at least two rows")
        if not np.isfinite(x).all() or not np.isfinite(dx).all():
            raise ValueError("Isolation Forest training data must be finite")
        self.x_med_, self.x_scale_ = robust_location_scale(x)
        self.dx_med_, self.dx_scale_ = robust_location_scale(dx)
        self.x_ref = valid_references(x)
        self.dx_ref = self.dx_med_.copy()
        self.n_features_in_ = x.shape[1]
        z = self._join(x, dx)
        self.model_ = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=min(self.max_samples, len(z)),
            contamination="auto",
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        ).fit(z)
        return self

    def _join(self, x: np.ndarray, dx: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        dx = np.asarray(dx, dtype=float)
        if x.ndim != 2 or x.shape != dx.shape or x.shape[1] != self.n_features_in_:
            raise ValueError("x and dx must be same-shaped arrays with the fitted feature count")
        if not np.isfinite(x).all() or not np.isfinite(dx).all():
            raise ValueError("Isolation Forest input must be finite")
        zx = np.clip((x - self.x_med_) / self.x_scale_, -1e4, 1e4)
        zdx = np.clip((dx - self.dx_med_) / self.dx_scale_, -1e4, 1e4)
        return np.concatenate((zx, zdx), axis=1)

    def score(self, x: np.ndarray, dx: np.ndarray) -> np.ndarray:
        """Return higher-is-more-anomalous scores, negating sklearn score_samples."""
        return -self.model_.score_samples(self._join(x, dx))

    def standardized(self, x: np.ndarray, dx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=float)
        dx = np.asarray(dx, dtype=float)
        if x.ndim != 2 or x.shape != dx.shape or x.shape[1] != self.n_features_in_:
            raise ValueError("x and dx must be same-shaped arrays with the fitted feature count")
        return (
            np.clip((x - self.x_med_) / self.x_scale_, -1e4, 1e4),
            np.clip((dx - self.dx_med_) / self.dx_scale_, -1e4, 1e4),
        )

    def model_bytes(self) -> int:
        """Approximate serialized tree-state size, excluding Python object overhead."""
        return int(sum(
            tree.tree_.__getstate__()["nodes"].nbytes
            + tree.tree_.__getstate__()["values"].nbytes
            for tree in self.model_.estimators_
        ))

