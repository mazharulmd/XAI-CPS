"""Forecasting-based anomaly detectors (trained on normal data only).

Every detector maps a window W in R^{w x d} to a one-step forecast x_hat = x_{t-1} + g(W)
(persistence plus a learnt correction). The anomaly score of a segment
Z = [W; x_t] in R^{(w+1) x d} is  e(Z) = mean_j ((x_t,j - x_hat_j) / sigma_j)^2,
where sigma_j is the validation residual std. `segment_score` is the single entry point
used by every explainer, so all explainers query exactly the same function.
"""
from __future__ import annotations

import time
import numpy as np
import torch
import torch.nn as nn


SIGMA_FLOOR = 0.05     # residual-scale floor, in train z-units (piecewise-constant SP tags)
CLIP = 20.0            # soft clip of normalised residuals


def soft_clip(r):
    return CLIP * torch.tanh(r / CLIP)


class Forecaster(nn.Module):
    def __init__(self, kind, d, w, hidden=64, lags=4):
        super().__init__()
        self.kind, self.d, self.w, self.lags = kind, d, w, lags
        if kind == "linear":
            self.net = nn.Linear(lags * d, d)
        elif kind == "mlp":
            self.net = nn.Sequential(nn.Flatten(), nn.Linear(w * d, 4 * hidden), nn.ReLU(),
                                     nn.Linear(4 * hidden, 2 * hidden), nn.ReLU(), nn.Linear(2 * hidden, d))
        elif kind == "cnn":
            self.conv = nn.Sequential(nn.Conv1d(d, hidden, 5, padding=2), nn.ReLU(),
                                      nn.Conv1d(hidden, hidden, 5, padding=2, dilation=1), nn.ReLU(),
                                      nn.Conv1d(hidden, hidden, 3, padding=2, dilation=2), nn.ReLU())
            self.head = nn.Linear(2 * hidden, d)
        elif kind == "gru":
            self.rnn = nn.GRU(d, hidden, batch_first=True)
            self.head = nn.Linear(hidden, d)
        else:
            raise ValueError(kind)

    def forward(self, W):                       # W: [B, w, d]
        last = W[:, -1]
        if self.kind == "linear":
            return last + self.net(W[:, -self.lags:].reshape(len(W), -1))
        if self.kind == "mlp":
            return last + self.net(W)
        if self.kind == "cnn":
            h = self.conv(W.transpose(1, 2))
            return last + self.head(torch.cat([h[:, :, -1], h.mean(-1)], 1))
        h, _ = self.rnn(W)
        return last + self.head(h[:, -1])


def _windows(arrs, w, stride):
    idx = []
    for ai, a in enumerate(arrs):
        for t in range(w, len(a), stride):
            idx.append((ai, t))
    return idx


def _batch(arrs, idx, w):
    W = np.stack([arrs[a][t - w:t] for a, t in idx])
    y = np.stack([arrs[a][t] for a, t in idx])
    return torch.from_numpy(W), torch.from_numpy(y)


class Detector:
    """Forecaster + residual scaling + EWMA alarm threshold (all fit on normal data)."""

    def __init__(self, kind, d, w=32, seed=0, hidden=64, alpha=0.1, q=0.999):
        torch.manual_seed(seed); np.random.seed(seed)
        self.kind, self.d, self.w, self.seed, self.alpha, self.q = kind, d, w, seed, alpha, q
        self.model = Forecaster(kind, d, w, hidden)
        self.sigma = torch.ones(d)
        self.theta = None
        self.train_info = {}

    # -------------------------------------------------------------- training
    def fit(self, train, val, epochs=12, batch=256, lr=1e-3, stride=2, max_windows=120_000, patience=2):
        rng = np.random.default_rng(self.seed)
        tr_idx = _windows(train, self.w, stride)
        va_idx = _windows(val, self.w, 7)
        va_idx = [va_idx[i] for i in rng.choice(len(va_idx), min(len(va_idx), 20_000), replace=False)]
        t0 = time.time()
        if self.kind == "linear":                                  # closed-form ridge
            sel = [tr_idx[i] for i in rng.choice(len(tr_idx), min(len(tr_idx), 200_000), replace=False)]
            W, y = _batch(train, sel, self.w)
            Xl = W[:, -self.model.lags:].reshape(len(W), -1).double()
            Y = (y - W[:, -1]).double()
            A = Xl.T @ Xl + 1.0 * torch.eye(Xl.shape[1], dtype=torch.float64)
            B = torch.linalg.solve(A, Xl.T @ Y)
            with torch.no_grad():
                self.model.net.weight.copy_(B.T.float()); self.model.net.bias.zero_()
        else:
            opt = torch.optim.Adam(self.model.parameters(), lr=lr)
            best, bad, best_state = np.inf, 0, None
            for ep in range(epochs):
                self.model.train()
                sel = rng.choice(len(tr_idx), min(len(tr_idx), max_windows), replace=False)
                for b in range(0, len(sel), batch):
                    W, y = _batch(train, [tr_idx[i] for i in sel[b:b + batch]], self.w)
                    loss = ((self.model(W) - y) ** 2).mean()
                    opt.zero_grad(); loss.backward(); opt.step()
                vl = self._val_loss(val, va_idx)
                if vl < best - 1e-5:
                    best, bad = vl, 0
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                else:
                    bad += 1
                    if bad >= patience:
                        break
            self.model.load_state_dict(best_state)
        self.model.eval()
        self.calibrate(val, va_idx)
        self.train_info = dict(train_time_s=time.time() - t0, val_mse=self._val_mse,
                               theta=self.theta, n_params=sum(p.numel() for p in self.model.parameters()))
        return self

    def calibrate(self, val, va_idx=None):
        """Residual scale and alarm threshold from VALIDATION normal data only (model frozen)."""
        if va_idx is None:
            va_idx = _windows(val, self.w, 7)
            rng = np.random.default_rng(self.seed)
            va_idx = [va_idx[i] for i in rng.choice(len(va_idx), min(len(va_idx), 20_000), replace=False)]
        res = []
        with torch.no_grad():
            for b in range(0, len(va_idx), 4096):
                W, y = _batch(val, va_idx[b:b + 4096], self.w)
                res.append((y - self.model(W)).numpy())
        res = np.concatenate(res)
        self.sigma = torch.from_numpy(np.maximum(res.std(0), SIGMA_FLOOR).astype(np.float32))
        s_val = np.concatenate([self.score_series(v) for v in val])
        s_val = s_val[np.isfinite(s_val)]
        self.theta = float(np.quantile(s_val, self.q))
        self._val_mse = float((res ** 2).mean())
        return self

    def _val_loss(self, val, idx):
        self.model.eval(); tot = 0.0
        with torch.no_grad():
            for b in range(0, len(idx), 4096):
                W, y = _batch(val, idx[b:b + 4096], self.w)
                tot += ((self.model(W) - y) ** 2).mean().item() * len(W)
        return tot / len(idx)

    # -------------------------------------------------------------- scoring
    def segment_score(self, Z):
        """Z: tensor [B, w+1, d] -> e(Z) [B] (differentiable)."""
        pred = self.model(Z[:, :-1])
        return (soft_clip((Z[:, -1] - pred) / self.sigma) ** 2).mean(-1)

    def residuals(self, X, batch=4096):
        """Per-tag normalised residuals r_t for t >= w (NaN before)."""
        n = len(X)
        R = np.full((n, self.d), np.nan, np.float32)
        Xt = torch.from_numpy(X)
        with torch.no_grad():
            for b in range(self.w, n, batch):
                e = min(n, b + batch)
                W = Xt[b - self.w:e - 1].unfold(0, self.w, 1).transpose(1, 2)   # [e-b, w, d]
                R[b:e] = soft_clip((Xt[b:e] - self.model(W)) / self.sigma).numpy()
        return R

    def score_series(self, X, R=None):
        R = self.residuals(X) if R is None else R
        e = (R ** 2).mean(1)
        s = np.full_like(e, np.nan)
        acc = None
        for t in range(len(e)):
            if np.isnan(e[t]):
                continue
            acc = e[t] if acc is None else self.alpha * e[t] + (1 - self.alpha) * acc
            s[t] = acc
        return s

    def save(self, path):
        torch.save(dict(kind=self.kind, d=self.d, w=self.w, seed=self.seed, alpha=self.alpha, q=self.q,
                        state=self.model.state_dict(), sigma=self.sigma, theta=self.theta,
                        info=self.train_info), path)

    @classmethod
    def load(cls, path):
        z = torch.load(path, weights_only=False)
        det = cls(z["kind"], z["d"], z["w"], z["seed"], alpha=z["alpha"], q=z["q"])
        det.model.load_state_dict(z["state"]); det.model.eval()
        det.sigma, det.theta, det.train_info = z["sigma"], z["theta"], z["info"]
        return det
