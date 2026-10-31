from __future__ import annotations

import numpy as np


def robust_location_scale(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    med = np.median(x, axis=0)
    mad = np.median(np.abs(x - med), axis=0) * 1.4826
    std = np.std(x, axis=0)
    scale = np.where(mad > 1e-8, mad, np.where(std > 1e-8, std, 1.0))
    return med, scale


def valid_references(x: np.ndarray, max_unique: int = 16) -> np.ndarray:
    """Use training medians for continuous channels and observed modes for low-cardinality channels."""
    refs=np.median(x,axis=0)
    for j in range(x.shape[1]):
        values,counts=np.unique(x[:,j],return_counts=True)
        if 1 < len(values) <= max_unique:
            refs[j]=values[int(np.argmax(counts))]
    return refs

class RobustZDetector:
    """Per-channel robust level and first-difference score baseline."""
    name = "RobustZ"

    def fit(self, x: np.ndarray, dx: np.ndarray) -> "RobustZDetector":
        self.x_med, self.x_scale = robust_location_scale(x)
        self.dx_med, self.dx_scale = robust_location_scale(dx)
        self.x_ref = valid_references(x)
        self.dx_ref = self.dx_med.copy()
        return self

    def transform(self, x: np.ndarray, dx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        zx = np.clip((x - self.x_med) / self.x_scale, -1e4, 1e4)
        zd = np.clip((dx - self.dx_med) / self.dx_scale, -1e4, 1e4)
        return zx, zd

    def score(self, x: np.ndarray, dx: np.ndarray) -> np.ndarray:
        zx, zd = self.transform(x, dx)
        per_channel = 0.5 * (np.abs(zx) + np.abs(zd))
        k = min(5, per_channel.shape[1])
        return np.partition(per_channel, -k, axis=1)[:, -k:].mean(axis=1)

    def contributions(self, x: np.ndarray, dx: np.ndarray) -> np.ndarray:
        zx, zd = self.transform(x, dx)
        return 0.5 * (np.abs(zx) + np.abs(zd))

    def standardized(self, x: np.ndarray, dx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.transform(x, dx)

class PCAResidualDetector:
    """PCA reconstruction residual over robustly standardized state and delta."""
    def __init__(self, variance: float = 0.95, max_components: int = 48):
        self.variance = variance
        self.max_components = max_components
        self.name = f"PCA-{int(variance*100)}"

    def fit(self, x: np.ndarray, dx: np.ndarray) -> "PCAResidualDetector":
        self.x_med, self.x_scale = robust_location_scale(x)
        self.dx_med, self.dx_scale = robust_location_scale(dx)
        self.x_ref = valid_references(x)
        self.dx_ref = self.dx_med.copy()
        z = self._join(x, dx)
        self.center = z.mean(axis=0)
        z = z - self.center
        cov = (z.T @ z) / max(1, len(z) - 1)
        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)[::-1]
        vals, vecs = np.maximum(vals[order], 0.0), vecs[:, order]
        total = max(float(vals.sum()), 1e-12)
        n = int(np.searchsorted(np.cumsum(vals) / total, self.variance) + 1)
        n = max(1, min(n, self.max_components, vecs.shape[1]))
        self.components = vecs[:, :n]
        self.n_components = n
        return self

    def _join(self, x: np.ndarray, dx: np.ndarray) -> np.ndarray:
        zx = np.clip((x - self.x_med) / self.x_scale, -1e3, 1e3)
        zd = np.clip((dx - self.dx_med) / self.dx_scale, -1e3, 1e3)
        return np.concatenate([zx, zd], axis=1)

    def residual(self, x: np.ndarray, dx: np.ndarray) -> np.ndarray:
        z = self._join(x, dx) - self.center
        projection = z @ self.components
        return z - projection @ self.components.T

    def score(self, x: np.ndarray, dx: np.ndarray) -> np.ndarray:
        r = self.residual(x, dx)
        return np.mean(r * r, axis=1)

    def contributions(self, x: np.ndarray, dx: np.ndarray) -> np.ndarray:
        r = self.residual(x, dx)
        d = x.shape[1]
        return r[:, :d] ** 2 + r[:, d:] ** 2

    def standardized(self, x: np.ndarray, dx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        zx = np.clip((x - self.x_med) / self.x_scale, -1e3, 1e3)
        zd = np.clip((dx - self.dx_med) / self.dx_scale, -1e3, 1e3)
        return zx, zd
