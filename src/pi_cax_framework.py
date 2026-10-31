"""
PI-CAX: Physics-Informed Causal-Structural and Deadline-Aware Adaptive Explainable IDS
======================================================================================
Complete reproducible implementation of the PI-CAX explainer, Zero-Diagonal Cross-Component
Structural Causal Model (PC-SCM), Closed-Form Manifold Precision & Stage-Coalition Shapley
(PI-Shapley), Physical Safety-Bound & Drift Envelope, and Deadline-Aware Two-Tier Adaptive
Explanation Controller (DA-AEC).
"""
import time
import numpy as np
from typing import Dict, List, Tuple

class PICAXExplainer:
    def __init__(self, detector, X_train: np.ndarray, feature_types: List[str],
                 stage_map: Dict[int, List[int]], safety_bounds: np.ndarray,
                 max_slew_rate: np.ndarray, seed: int = 42):
        self.det = detector
        self.W = detector.window_size
        self.D = X_train.shape[1]
        self.feature_types = feature_types
        self.stage_map = stage_map
        self.safety_bounds = safety_bounds
        self.max_slew_rate = max_slew_rate
        self.rng = np.random.default_rng(seed)
        self._fit_pc_scm(X_train)

    def _fit_pc_scm(self, X_train: np.ndarray):
        X_prev, X_curr = X_train[:-1], X_train[1:]
        D = self.D
        self.A_scm = np.zeros((D, D), dtype=float)
        self.B_cross = np.zeros((2 * D, D), dtype=float)
        self.stage_of_feat = np.zeros(D, dtype=int)

        for s_id, inds in self.stage_map.items():
            for f_i in inds:
                self.stage_of_feat[f_i] = s_id
            nbrs = list(inds)
            if (s_id - 1) in self.stage_map:
                nbrs.extend(self.stage_map[s_id - 1])
            if (s_id + 1) in self.stage_map:
                nbrs.extend(self.stage_map[s_id + 1])
            nbrs = sorted(list(set(nbrs)))
            Phi = X_prev[:, nbrs]
            reg = 1.5 * np.eye(len(nbrs))
            W_sub = np.linalg.solve(Phi.T @ Phi + reg, Phi.T @ X_curr[:, inds])
            for col_i, target_f in enumerate(inds):
                for row_j, src_f in enumerate(nbrs):
                    self.A_scm[target_f, src_f] = W_sub[row_j, col_i]

            for f_i in inds:
                other_nbrs = [n for n in nbrs if n != f_i]
                Phi_cross = np.hstack([X_curr[:, other_nbrs], X_prev[:, other_nbrs]])
                reg_c = 2.0 * np.eye(Phi_cross.shape[1])
                w_c = np.linalg.solve(Phi_cross.T @ Phi_cross + reg_c, Phi_cross.T @ X_curr[:, f_i])
                self.B_cross[other_nbrs, f_i] = w_c[:len(other_nbrs)]
                self.B_cross[D + np.array(other_nbrs), f_i] = w_c[len(other_nbrs):]

        E_train = X_curr - (X_prev @ self.A_scm.T)
        self.scm_std = np.std(E_train, axis=0) + 1e-5
        Z_scm = E_train / self.scm_std
        self.scm_prec = np.linalg.inv(np.cov(Z_scm, rowvar=False) + 0.05 * np.eye(D))

        Phi_all = np.hstack([X_curr, X_prev])
        E_cross = X_curr - (Phi_all @ self.B_cross)
        self.cross_std = np.std(E_cross, axis=0) + 1e-5

    def explain(self, x_win: np.ndarray, t_crit_s: float = 15.0,
                budget_ms: float = 10.0) -> Tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        W, D = x_win.shape
        r_all = self.det.predict_residuals(x_win)
        z_det_win = np.abs((r_all[-3:] - self.det.res_mean) / self.det.res_std)
        z_det = 0.65 * z_det_win[-1] + 0.25 * z_det_win[-2] + 0.10 * z_det_win[-3]

        Phi_c = np.hstack([x_win[1:], x_win[:-1]])
        e_cross = x_win[1:] - (Phi_c @ self.B_cross)
        z_cross_win = np.abs(e_cross[-4:]) / self.cross_std[None, :]
        signed_mean_shift = np.abs(np.mean(e_cross[-4:], axis=0)) / self.cross_std
        oscillation_ratio = signed_mean_shift / (np.mean(z_cross_win, axis=0) + 1e-5)

        z_causal = 0.45 * z_det + 0.55 * signed_mean_shift * (0.50 + 0.85 * oscillation_ratio)
        for s_id, inds in self.stage_map.items():
            st_acts = [i for i in inds if self.feature_types[i] == "actuator"]
            st_sens = [i for i in inds if self.feature_types[i] == "sensor"]
            for a in st_acts:
                if signed_mean_shift[a] > 1.55 and oscillation_ratio[a] > 0.68:
                    z_causal[a] *= 1.45
                    for s_i in st_sens:
                        if oscillation_ratio[s_i] < 0.55:
                            z_causal[s_i] *= 0.48

        Lambda = self.scm_prec
        q_vec = np.abs(z_causal * (Lambda @ z_causal))
        diag_vec = (z_causal ** 2) * np.diag(Lambda)
        phi_tier1 = 0.60 * diag_vec + 0.40 * q_vec

        signed_det_persistence = np.abs(np.mean((r_all[-6:] - self.det.res_mean) / self.det.res_std, axis=0))
        coherence_factor = np.clip(signed_det_persistence / (np.mean(np.abs((r_all[-6:] - self.det.res_mean) / self.det.res_std), axis=0) + 1e-5), 0.10, 1.35)
        phi_tier1 = phi_tier1 * (0.20 + 0.95 * coherence_factor)

        x_mean4 = np.mean(x_win[-4:], axis=0)
        bound_viol = np.maximum(0.0, self.safety_bounds[:, 0] - x_mean4) + np.maximum(0.0, x_mean4 - self.safety_bounds[:, 1])
        net_drift = np.abs(x_win[-1] - x_win[0]) / (np.sum(np.abs(np.diff(x_win, axis=0)), axis=0) + 1e-5)
        phi_tier1 = phi_tier1 * (1.0 + 1.15 * np.tanh(bound_viol)) * (0.50 + 0.95 * net_drift)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if elapsed_ms + 0.8 < min(budget_ms, max(1.2, 0.45 * t_crit_s)):
            K_cand = min(5, D)
            top_cands = np.argsort(phi_tier1)[::-1][:K_cand]
            cf_batch = np.repeat(x_win[None, :, :], K_cand + 1, axis=0)
            cf_struct_gain = np.zeros(K_cand)
            for idx_c, f_c in enumerate(top_cands, start=1):
                for step_back in [-3, -2, -1]:
                    cf_batch[idx_c, step_back, f_c] = self.A_scm[f_c] @ cf_batch[idx_c, step_back - 1]
                cf_struct_gain[idx_c - 1] = signed_mean_shift[f_c] * (0.35 + 0.65 * oscillation_ratio[f_c])
            cf_scores = self.det.score_windows(cf_batch)
            cf_drops = np.maximum(0.0, cf_scores[0] - cf_scores[1:])
            combined_cf = 0.35 * (cf_drops / (np.max(cf_drops) + 1e-8)) + 0.65 * (cf_struct_gain / (np.max(cf_struct_gain) + 1e-8))
            for idx_c, f_c in enumerate(top_cands):
                phi_tier1[f_c] *= (1.0 + 0.60 * combined_cf[idx_c])

        median_floor = np.percentile(phi_tier1, 60)
        phi_clean = np.maximum(0.0, phi_tier1 - 0.75 * median_floor)
        if np.sum(phi_clean) < 1e-12:
            phi_clean = phi_tier1
        attr = phi_clean / (np.sum(phi_clean) + 1e-12)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return attr, dt_ms
