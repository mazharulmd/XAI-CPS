"""PI-CAX explainer, ported for evaluation on real data under our protocol.

PI-CAX ("physics-informed causal-structural and deadline-aware adaptive explainability") was
designed and tuned in a parallel exploratory study of this project on synthetic multi-stage plant
models (branch feat/pi-cax-q1-manuscript, src/explainers.py::explain_pi_cax). This port keeps its
equations and every numeric constant unchanged, so that the evaluation tests whether the published
design transfers to real data. Adaptations, all forced by the interface:

* stages        -> our control loops (same order as configs; neighbour stages = adjacent loops);
* residuals     -> our forecaster's soft-clipped normalised one-step residuals, standardised with
                   training-residual mean/std as in the original;
* score_windows -> our segment score e(.) on the full detector segment, with the counterfactual
                   edit applied to its last three rows;
* safety bounds -> per-tag min/max over normal training data (the synthetic plants had explicit
                   bounds; none exist for HAI);
* tier-2 gate   -> always on. The original skips tier 2 when tier 1 exceeds a wall-clock budget,
                   which makes outputs machine-dependent; tier 2 only adds the counterfactual check.
"""
import numpy as np
import torch

from .detectors import soft_clip

W_PICAX = 10           # window_size of the original configuration


class PICAX:
    def __init__(self, det, train, groups, actuator, max_rows=400_000, seed=0):
        self.det = det
        self.groups = [np.asarray(g) for g in groups]
        self.D = train[0].shape[1]
        self.stage_map = {i: list(map(int, g)) for i, g in enumerate(self.groups)}
        self.feature_types = ["actuator" if a else "sensor" for a in actuator]
        cat = np.concatenate(train, 0)
        self.safety_bounds = np.stack([cat.min(0), cat.max(0)], 1)
        prev = np.concatenate([a[:-1] for a in train], 0)
        curr = np.concatenate([a[1:] for a in train], 0)
        if len(prev) > max_rows:                               # deterministic thinning for the ridge fits
            idx = np.linspace(0, len(prev) - 1, max_rows).astype(int)
            prev, curr = prev[idx], curr[idx]
        self._fit_pc_scm(prev.astype(np.float64), curr.astype(np.float64))
        self._fit_residual_stats(train, seed)

    # ---- verbatim port of CPSExplainerSuite._fit_pc_scm (X_prev/X_curr built per file)
    def _fit_pc_scm(self, X_prev, X_curr):
        D = self.D
        self.A_scm = np.zeros((D, D))
        self.B_cross = np.zeros((2 * D, D))
        for s_id, inds in self.stage_map.items():
            nbrs = list(inds)
            if (s_id - 1) in self.stage_map:
                nbrs.extend(self.stage_map[s_id - 1])
            if (s_id + 1) in self.stage_map:
                nbrs.extend(self.stage_map[s_id - 1] if (s_id - 1) in self.stage_map else [])
                nbrs.extend(self.stage_map[s_id + 1])
            nbrs = sorted(set(nbrs))
            Phi = X_prev[:, nbrs]
            W_sub = np.linalg.solve(Phi.T @ Phi + 1.5 * np.eye(len(nbrs)), Phi.T @ X_curr[:, inds])
            for col_i, target_f in enumerate(inds):
                for row_j, src_f in enumerate(nbrs):
                    self.A_scm[target_f, src_f] = W_sub[row_j, col_i]
            for f_i in inds:
                other = [n for n in nbrs if n != f_i]
                Phi_c = np.hstack([X_curr[:, other], X_prev[:, other]])
                w_c = np.linalg.solve(Phi_c.T @ Phi_c + 2.0 * np.eye(Phi_c.shape[1]), Phi_c.T @ X_curr[:, f_i])
                self.B_cross[other, f_i] = w_c[:len(other)]
                self.B_cross[D + np.array(other), f_i] = w_c[len(other):]
        E = X_curr - X_prev @ self.A_scm.T
        self.scm_std = E.std(0) + 1e-5
        cov = np.cov(E / self.scm_std, rowvar=False) + 0.05 * np.eye(D)
        self.scm_prec = np.linalg.inv(cov)
        Phi_c = np.hstack([X_curr, X_prev])
        self.cross_std = (X_curr - Phi_c @ self.B_cross).std(0) + 1e-5

    def recalibrate(self, val, seed=0):
        """Refit the normalising statistics on (noisy) normal validation data, ridge models frozen.

        Counterpart of Detector.calibrate for the recalibrated-noise condition: residual mean/std,
        loop-model residual scale/precision and cross-model residual scale are re-estimated.
        """
        prev = np.concatenate([a[:-1] for a in val], 0).astype(np.float64)
        curr = np.concatenate([a[1:] for a in val], 0).astype(np.float64)
        E = curr - prev @ self.A_scm.T
        self.scm_std = E.std(0) + 1e-5
        self.scm_prec = np.linalg.inv(np.cov(E / self.scm_std, rowvar=False) + 0.05 * np.eye(self.D))
        self.cross_std = (curr - np.hstack([curr, prev]) @ self.B_cross).std(0) + 1e-5
        self._fit_residual_stats(val, seed)
        return self

    def _residuals(self, X, ts):
        """Normalised one-step residuals of our forecaster at times ts (numpy [len(ts), d])."""
        w = self.det.w
        with torch.no_grad():
            Z = torch.stack([X[t - w:t + 1] for t in ts])
            pred = self.det.model(Z[:, :-1])
            return soft_clip((Z[:, -1] - pred) / self.det.sigma).numpy().astype(np.float64)

    def _fit_residual_stats(self, train, seed, n=20_000):
        rng = np.random.default_rng(seed)
        rs = []
        for a in train:
            X = torch.from_numpy(a)
            ts = rng.choice(np.arange(self.det.w, len(a)), min(n // len(train), len(a) - self.det.w), replace=False)
            for b in range(0, len(ts), 4096):
                rs.append(self._residuals(X, ts[b:b + 4096]))
        r = np.concatenate(rs)
        self.res_mean, self.res_std = r.mean(0), r.std(0) + 1e-5

    # ---- verbatim port of explain_pi_cax (all four components enabled)
    def explain(self, ctx, t, causal=True, manifold=True, envelope=True, tier2=True):
        """Flags mirror the original's ablation switches (use_causal_decoupling, use_manifold_shapley, ...)."""
        X = ctx.X
        x_win = X[t - W_PICAX + 1:t + 1].numpy().astype(np.float64)
        r_all = self._residuals(X, range(t - W_PICAX + 1, t + 1))
        z = lambda r: (r - self.res_mean) / self.res_std
        z_det_win = np.abs(z(r_all[-3:]))
        z_det = 0.65 * z_det_win[-1] + 0.25 * z_det_win[-2] + 0.10 * z_det_win[-3]

        Phi_c = np.hstack([x_win[1:], x_win[:-1]])
        e_cross = x_win[1:] - Phi_c @ self.B_cross
        z_cross_win = np.abs(e_cross[-4:]) / self.cross_std[None, :]
        shift = np.abs(e_cross[-4:].mean(0)) / self.cross_std
        osc = shift / (z_cross_win.mean(0) + 1e-5)

        zc = 0.45 * z_det + 0.55 * shift * (0.50 + 0.85 * osc) if causal else z_det.copy()
        for inds in (self.stage_map.values() if causal else []):
            acts = [i for i in inds if self.feature_types[i] == "actuator"]
            sens = [i for i in inds if self.feature_types[i] == "sensor"]
            for a in acts:
                if shift[a] > 1.55 and osc[a] > 0.68:
                    zc[a] *= 1.45
                    for s_i in sens:
                        if osc[s_i] < 0.55:
                            zc[s_i] *= 0.48

        Lam = self.scm_prec
        phi = zc ** 2 if not manifold else 0.60 * (zc ** 2) * np.diag(Lam) + 0.40 * np.abs(zc * (Lam @ zc))
        if manifold:
            zr = z(r_all[-6:])
            coh = np.clip(np.abs(zr.mean(0)) / (np.abs(zr).mean(0) + 1e-5), 0.10, 1.35)
            phi = phi * (0.20 + 0.95 * coh)
            energy = {}
            for s_id, inds in self.stage_map.items():
                active = np.sum(z_det[inds] > 1.45) / float(len(inds))
                energy[s_id] = float(np.mean(phi[inds])) * (0.40 + 0.90 * active)
            mx = max(energy.values()) + 1e-9
            for s_id, inds in self.stage_map.items():
                phi[inds] *= 0.35 + 0.65 * (energy[s_id] / mx) ** 0.75

        x4 = x_win[-4:].mean(0)
        viol = np.maximum(0.0, self.safety_bounds[:, 0] - x4) + np.maximum(0.0, x4 - self.safety_bounds[:, 1])
        drift = np.abs(x_win[-1] - x_win[0]) / (np.abs(np.diff(x_win, axis=0)).sum(0) + 1e-5)
        if envelope:
            phi = phi * (1.0 + 1.15 * np.tanh(viol)) * (0.50 + 0.95 * drift)

        # tier 2: counterfactual check of the top-5 candidates on the detector score
        K = min(5, self.D) if tier2 else 0
        top = np.argsort(phi)[::-1][:K]
        seg = ctx.seg(t).numpy().astype(np.float64)
        batch = np.repeat(seg[None], K + 1, 0)
        gain = np.zeros(K)
        for c, f in enumerate(top, start=1):
            for sb in (-3, -2, -1):
                batch[c, sb, f] = self.A_scm[f] @ batch[c, sb - 1]
            gain[c - 1] = shift[f] * (0.35 + 0.65 * osc[f])
        with torch.no_grad():
            sc = ctx.det.segment_score(torch.from_numpy(batch.astype(np.float32))).numpy().astype(np.float64)
        if K:
            drops = np.maximum(0.0, sc[0] - sc[1:])
            comb = 0.35 * drops / (drops.max() + 1e-8) + 0.65 * gain / (gain.max() + 1e-8)
            for c, f in enumerate(top):
                phi[f] *= 1.0 + 0.60 * comb[c]

        clean = np.maximum(0.0, phi - 0.75 * np.percentile(phi, 60))
        if clean.sum() < 1e-12:
            clean = phi
        tag = clean / (clean.sum() + 1e-12)
        return dict(tag=tag, loop=ctx.to_loops(tag), evals=W_PICAX + K + 1)
