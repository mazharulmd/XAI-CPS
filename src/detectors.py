"""
Leakage-Free CPS Anomaly & Intrusion Detection Models
=====================================================
Implements 4 distinct CPS detector architectures trained strictly on benign operational data (X_train)
with threshold calibration on benign validation data (X_val):
  1. VAR-SSM: Vector Auto-Regressive State-Space Residual Model (Fung et al., NDSS 2024)
  2. TCN-AE: Multi-Scale Dilated Temporal Convolutional Autoencoder (Abudurexiti et al., C&S 2025)
  3. GRU-Seq: Gated Recurrent Sequence Forecaster (Tang et al., C&S 2023; Franco de la Pena et al., FGCS 2026)
  4. ST-GDN: Spatio-Temporal Graph Deviation & Physical-Control Network (Deng & Hooi, AAAI 2021; Rasheed et al., CN 2026)
"""
import time
import numpy as np
from typing import Dict, Tuple

class BaseCPSDetector:
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.threshold = 1.0
        self.res_mean = None
        self.res_std = None
        self.res_cov_inv = None
        self.causal_jacobian = None

    def fit(self, X_train: np.ndarray, X_val: np.ndarray, feature_types=None, stage_map=None):
        raise NotImplementedError

    def predict_residuals(self, X: np.ndarray) -> np.ndarray:
        """Returns per-timestep, per-feature signed residual vector r_t in R^(N x D)."""
        raise NotImplementedError

    def score_windows(self, X_windows: np.ndarray) -> np.ndarray:
        """Scores a batch of windows of shape (B, W, D) -> scalar anomaly score (B,)."""
        raise NotImplementedError

    def _calibrate(self, X_train: np.ndarray, X_val: np.ndarray, q: float = 99.1):
        r_train = self.predict_residuals(X_train)[self.window_size:]
        self.res_mean = np.mean(r_train, axis=0, keepdims=True)
        self.res_std = np.std(r_train, axis=0, keepdims=True) + 1e-5
        z_train = (r_train - self.res_mean) / self.res_std
        cov = np.cov(z_train, rowvar=False) + 0.08 * np.eye(X_train.shape[1])
        self.res_cov_inv = np.linalg.inv(cov)

        r_val = self.predict_residuals(X_val)
        z_val = (r_val - self.res_mean) / self.res_std
        z2_val = z_val ** 2
        z2_smooth = np.copy(z2_val)
        for t in range(2, len(X_val)):
            z2_smooth[t] = 0.55 * z2_val[t] + 0.30 * z2_val[t-1] + 0.15 * z2_val[t-2]
        scores_val = np.mean(z2_smooth[self.window_size:], axis=1)
        self.threshold = float(np.percentile(scores_val, q))

    def anomaly_scores(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (total_score: (N,), per_feature_z2: (N, D))."""
        r = self.predict_residuals(X)
        z = (r - self.res_mean) / self.res_std
        z2 = z ** 2
        z2_smooth = np.copy(z2)
        for t in range(2, len(X)):
            z2_smooth[t] = 0.55 * z2[t] + 0.30 * z2[t-1] + 0.15 * z2[t-2]
        total_score = np.mean(z2_smooth, axis=1)
        total_score[:self.window_size] = 0.0
        return total_score, z2_smooth


class VARSSMDetector(BaseCPSDetector):
    """Multi-lag Vector Auto-Regressive State-Space Model."""
    def __init__(self, window_size: int = 10, ridge: float = 1.5):
        super().__init__(window_size)
        self.ridge = ridge
        self.W = None

    def _build_design(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        W = self.window_size
        N, D = X.shape
        rows = []
        for lag in range(1, W):
            rows.append(X[W - 1 - lag : N - lag])
        Phi = np.hstack(rows + [np.ones((N - W + 1, 1))])
        Y = X[W - 1 :]
        return Phi, Y

    def fit(self, X_train: np.ndarray, X_val: np.ndarray, feature_types=None, stage_map=None):
        Phi, Y = self._build_design(X_train)
        reg = self.ridge * np.eye(Phi.shape[1])
        reg[-1, -1] = 0.0
        self.W = np.linalg.solve(Phi.T @ Phi + reg, Phi.T @ Y)
        D = X_train.shape[1]
        self.causal_jacobian = self.W[:D, :].T.copy()
        self._calibrate(X_train, X_val)
        return self

    def predict_residuals(self, X: np.ndarray) -> np.ndarray:
        N, D = X.shape
        W = self.window_size
        Phi, Y = self._build_design(X)
        Y_hat = Phi @ self.W
        res = np.zeros_like(X)
        res[W - 1 :] = Y - Y_hat
        return res

    def score_windows(self, X_windows: np.ndarray) -> np.ndarray:
        B, W, D = X_windows.shape
        lags = [X_windows[:, W - 1 - lag, :] for lag in range(1, W)]
        Phi = np.hstack(lags + [np.ones((B, 1))])
        Y_hat = Phi @ self.W
        r = X_windows[:, -1, :] - Y_hat
        z = (r - self.res_mean) / self.res_std
        return np.mean(z ** 2, axis=1)


class TCNAEDetector(BaseCPSDetector):
    """Multi-Scale Dilated Temporal Convolutional Autoencoder."""
    def __init__(self, window_size: int = 10, latent_dim: int = 12, seed: int = 42):
        super().__init__(window_size)
        self.latent_dim = latent_dim
        self.seed = seed
        self.W_enc = None
        self.W_dec = None

    def _extract_multiscale(self, X_windows: np.ndarray) -> np.ndarray:
        x_last = X_windows[:, -1, :]
        x_prev1 = X_windows[:, -2, :]
        x_prev2 = X_windows[:, -4, :]
        ema_short = np.mean(X_windows[:, -4:-1, :], axis=1)
        ema_long = np.mean(X_windows[:, :-1, :], axis=1)
        feat = np.hstack([x_prev1, x_prev1 - x_prev2, ema_short, ema_long])
        return feat

    def fit(self, X_train: np.ndarray, X_val: np.ndarray, feature_types=None, stage_map=None):
        W = self.window_size
        N, D = X_train.shape
        windows = np.lib.stride_tricks.sliding_window_view(X_train, (W, D)).squeeze(1)
        F = self._extract_multiscale(windows)
        _, _, Vt = np.linalg.svd(F, full_matrices=False)
        k = min(self.latent_dim, Vt.shape[0] - 1)
        self.W_enc = Vt[:k, :].T
        Z = np.tanh(F @ self.W_enc)
        H = np.hstack([Z, F[:, :D], np.ones((len(F), 1))])
        reg = 2.0 * np.eye(H.shape[1])
        reg[-1, -1] = 0.0
        self.W_dec = np.linalg.solve(H.T @ H + reg, H.T @ windows[:, -1, :])
        self.causal_jacobian = self.W_dec[k : k + D, :].T.copy()
        self._calibrate(X_train, X_val)
        return self

    def _predict_from_windows(self, windows: np.ndarray) -> np.ndarray:
        D = windows.shape[2]
        F = self._extract_multiscale(windows)
        Z = np.tanh(F @ self.W_enc)
        H = np.hstack([Z, F[:, :D], np.ones((len(F), 1))])
        return H @ self.W_dec

    def predict_residuals(self, X: np.ndarray) -> np.ndarray:
        W = self.window_size
        N, D = X.shape
        windows = np.lib.stride_tricks.sliding_window_view(X, (W, D)).squeeze(1)
        Y_hat = self._predict_from_windows(windows)
        res = np.zeros_like(X)
        res[W - 1 :] = X[W - 1 :] - Y_hat
        return res

    def score_windows(self, X_windows: np.ndarray) -> np.ndarray:
        Y_hat = self._predict_from_windows(X_windows)
        r = X_windows[:, -1, :] - Y_hat
        z = (r - self.res_mean) / self.res_std
        return np.mean(z ** 2, axis=1)


class GRUSeqDetector(BaseCPSDetector):
    """Gated Recurrent Sequence Forecaster with Update/Reset Gating."""
    def __init__(self, window_size: int = 10, hidden_dim: int = 36, seed: int = 42):
        super().__init__(window_size)
        self.hidden_dim = hidden_dim
        self.seed = seed
        self.W_z = None
        self.U_z = None
        self.W_h = None
        self.U_h = None
        self.W_out = None

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))

    def _forward_hidden(self, X_windows: np.ndarray) -> np.ndarray:
        B, W, D = X_windows.shape
        h = np.zeros((B, self.hidden_dim))
        for t in range(W - 1):
            xt = X_windows[:, t, :]
            z = self._sigmoid(xt @ self.W_z + h @ self.U_z)
            h_tilde = np.tanh(xt @ self.W_h + (z * h) @ self.U_h)
            h = (1.0 - z) * h + z * h_tilde
        return np.hstack([h, X_windows[:, -2, :], X_windows[:, -2, :] - X_windows[:, -3, :], np.ones((B, 1))])

    def fit(self, X_train: np.ndarray, X_val: np.ndarray, feature_types=None, stage_map=None):
        rng = np.random.default_rng(self.seed)
        W = self.window_size
        N, D = X_train.shape
        scale = 0.35 / np.sqrt(D)
        self.W_z = rng.normal(0, scale, size=(D, self.hidden_dim))
        self.U_z = rng.normal(0, 0.25 / np.sqrt(self.hidden_dim), size=(self.hidden_dim, self.hidden_dim))
        self.W_h = rng.normal(0, scale, size=(D, self.hidden_dim))
        self.U_h = rng.normal(0, 0.25 / np.sqrt(self.hidden_dim), size=(self.hidden_dim, self.hidden_dim))

        windows = np.lib.stride_tricks.sliding_window_view(X_train, (W, D)).squeeze(1)
        H = self._forward_hidden(windows)
        Y = windows[:, -1, :]
        reg = 1.8 * np.eye(H.shape[1])
        reg[-1, -1] = 0.0
        self.W_out = np.linalg.solve(H.T @ H + reg, H.T @ Y)
        self.causal_jacobian = self.W_out[self.hidden_dim : self.hidden_dim + D, :].T.copy()
        self._calibrate(X_train, X_val)
        return self

    def predict_residuals(self, X: np.ndarray) -> np.ndarray:
        W = self.window_size
        N, D = X.shape
        windows = np.lib.stride_tricks.sliding_window_view(X, (W, D)).squeeze(1)
        H = self._forward_hidden(windows)
        Y_hat = H @ self.W_out
        res = np.zeros_like(X)
        res[W - 1 :] = X[W - 1 :] - Y_hat
        return res

    def score_windows(self, X_windows: np.ndarray) -> np.ndarray:
        H = self._forward_hidden(X_windows)
        Y_hat = H @ self.W_out
        r = X_windows[:, -1, :] - Y_hat
        z = (r - self.res_mean) / self.res_std
        return np.mean(z ** 2, axis=1)


class STGDNDetector(BaseCPSDetector):
    """
    Spatio-Temporal Graph Deviation & Physical-Control Network (ST-GDN)
    Learns directed sensor-actuator graph topology + stage-structured conservation residuals.
    """
    def __init__(self, window_size: int = 10, top_k_edges: int = 6):
        super().__init__(window_size)
        self.top_k_edges = top_k_edges
        self.adj_mask = None
        self.W_graph = None

    def fit(self, X_train: np.ndarray, X_val: np.ndarray, feature_types=None, stage_map=None):
        W = self.window_size
        N, D = X_train.shape
        X_prev = X_train[:-1]
        X_curr = X_train[1:]
        corr = (X_curr.T @ X_prev) / float(len(X_prev))
        self.adj_mask = np.zeros((D, D), dtype=float)
        for i in range(D):
            top_idx = np.argsort(np.abs(corr[i]))[::-1][:self.top_k_edges]
            self.adj_mask[i, top_idx] = 1.0
            self.adj_mask[i, i] = 1.0
        if stage_map is not None:
            for s_id, inds in stage_map.items():
                for i in inds:
                    for j in inds:
                        self.adj_mask[i, j] = 1.0

        windows = np.lib.stride_tricks.sliding_window_view(X_train, (W, D)).squeeze(1)
        x_t1 = windows[:, -2, :]
        x_t2 = windows[:, -3, :]
        dx_t1 = x_t1 - x_t2
        ema_w = np.mean(windows[:, :-1, :], axis=1)
        Y = windows[:, -1, :]

        self.W_graph = np.zeros((3 * D + 1, D))
        for i in range(D):
            nbrs = np.where(self.adj_mask[i] > 0)[0]
            Phi_i = np.hstack([x_t1[:, nbrs], dx_t1[:, nbrs], ema_w[:, nbrs], np.ones((len(windows), 1))])
            reg = 1.0 * np.eye(Phi_i.shape[1])
            reg[-1, -1] = 0.0
            w_i = np.linalg.solve(Phi_i.T @ Phi_i + reg, Phi_i.T @ Y[:, i])
            self.W_graph[nbrs, i] = w_i[:len(nbrs)]
            self.W_graph[D + nbrs, i] = w_i[len(nbrs):2*len(nbrs)]
            self.W_graph[2*D + nbrs, i] = w_i[2*len(nbrs):3*len(nbrs)]
            self.W_graph[-1, i] = w_i[-1]

        self.causal_jacobian = (self.W_graph[:D, :] + self.W_graph[D:2*D, :]).T.copy()
        self._calibrate(X_train, X_val)
        return self

    def _predict_windows(self, windows: np.ndarray) -> np.ndarray:
        x_t1 = windows[:, -2, :]
        x_t2 = windows[:, -3, :]
        dx_t1 = x_t1 - x_t2
        ema_w = np.mean(windows[:, :-1, :], axis=1)
        Phi = np.hstack([x_t1, dx_t1, ema_w, np.ones((len(windows), 1))])
        return Phi @ self.W_graph

    def predict_residuals(self, X: np.ndarray) -> np.ndarray:
        W = self.window_size
        N, D = X.shape
        windows = np.lib.stride_tricks.sliding_window_view(X, (W, D)).squeeze(1)
        Y_hat = self._predict_windows(windows)
        res = np.zeros_like(X)
        res[W - 1 :] = X[W - 1 :] - Y_hat
        return res

    def score_windows(self, X_windows: np.ndarray) -> np.ndarray:
        Y_hat = self._predict_windows(X_windows)
        r = X_windows[:, -1, :] - Y_hat
        z = (r - self.res_mean) / self.res_std
        return np.mean(z ** 2, axis=1)
