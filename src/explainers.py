"""
XAI Attribution Baselines & Proposed PI-CAX Framework for CPS Intrusion Detection
=================================================================================
Implements 6 state-of-the-art CPS XAI baselines and the proposed PI-CAX framework:
  1. Raw-MSE: Per-feature standardized residual energy (Fung et al., NDSS 2024)
  2. VG (Vanilla Gradient / Saliency): Finite-difference / analytical input gradient (Oinonen & Morsi, EPSR 2026)
  3. IG (Integrated Gradients): Path integral along straight-line interpolation from benign baseline (Fung et al., 2024; Mathuros et al., 2024)
  4. LIME: Local Ridge Surrogate over Gaussian-perturbed windows (Ribeiro et al., 2016; Oinonen & Morsi, 2026)
  5. KernelSHAP: Game-theoretic Shapley regression with post-hoc temporal aggregation (Lundberg & Lee, 2017; Franco de la Pena et al., 2026)
  6. ShaTS: A priori Sensor/Actuator & Process-Stage grouped Shapley values (Franco de la Pena et al., FGCS 2026)
  7. PI-CAX (Proposed): Physics-Informed Causal-Structural & Deadline-Aware Adaptive Explainability:
     - Component A (PC-SCM): Unsupervised Zero-Diagonal Cross-Component Physical Conservation & PLC Control-Law Decoupling
     - Component B (PI-Shapley): Closed-Form Manifold-Constrained Precision & Stage-Coalition Shapley Attribution
     - Component C (Phys-Envelope): Unsupervised Safety-Bound & Slew-Rate Physical Plausibility Modulation
     - Component D (DA-AEC): Deadline-Aware Two-Tier Adaptive Counterfactual Verification under T_crit budget
"""
import time
import numpy as np
from typing import Dict, List, Optional, Tuple

class CPSExplainerSuite:
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

        all_win = np.lib.stride_tricks.sliding_window_view(X_train, (self.W, self.D)).squeeze(1)
        bg_idx = np.linspace(0, len(all_win) - 1, 32, dtype=int)
        self.bg_windows = all_win[bg_idx].copy()
        self.bg_mean = np.mean(self.bg_windows, axis=0)

        self._fit_pc_scm(X_train)

    def _fit_pc_scm(self, X_train: np.ndarray):
        X_prev = X_train[:-1]
        X_curr = X_train[1:]
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
                nbrs.extend(self.stage_map[s_id - 1] if (s_id - 1) in self.stage_map else [])
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
        cov_scm = np.cov(Z_scm, rowvar=False) + 0.05 * np.eye(D)
        self.scm_cov = cov_scm
        self.scm_prec = np.linalg.inv(cov_scm)

        Phi_all = np.hstack([X_curr, X_prev])
        E_cross = X_curr - (Phi_all @ self.B_cross)
        self.cross_std = np.std(E_cross, axis=0) + 1e-5

        self.is_actuator = np.array([1.0 if t == "actuator" else 0.0 for t in self.feature_types])
        self.is_sensor = 1.0 - self.is_actuator

    def explain_raw_mse(self, x_win: np.ndarray) -> Tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        r = self.det.predict_residuals(x_win)[-1]
        z2 = ((r - self.det.res_mean.ravel()) / self.det.res_std.ravel()) ** 2
        attr = z2 / (np.sum(z2) + 1e-12)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return attr, dt_ms

    def explain_vg(self, x_win: np.ndarray, eps: float = 1e-2) -> Tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        W, D = x_win.shape
        K_steps = 3
        batch = np.repeat(x_win[None, :, :], K_steps * D + 1, axis=0)
        idx = 1
        for k in range(1, K_steps + 1):
            for j in range(D):
                batch[idx, W - k, j] += eps
                idx += 1
        scores = self.det.score_windows(batch)
        base_s = scores[0]
        grads = np.abs(scores[1:] - base_s).reshape(K_steps, D) / eps
        sal = np.mean(grads * np.abs(x_win[-K_steps:, :]), axis=0)
        attr = sal / (np.sum(sal) + 1e-12)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return attr, dt_ms

    def explain_ig(self, x_win: np.ndarray, m_steps: int = 25) -> Tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        W, D = x_win.shape
        diff = x_win - self.bg_mean
        alphas = np.linspace(0.05, 1.0, m_steps)
        eps = 1e-2
        total_grad = np.zeros(D)
        batch = np.zeros((m_steps * (D + 1), W, D))
        b_idx = 0
        for a in alphas:
            interp = self.bg_mean + a * diff
            batch[b_idx] = interp
            b_idx += 1
            for j in range(D):
                pert = interp.copy()
                pert[-1, j] += eps
                pert[-2, j] += 0.5 * eps
                batch[b_idx] = pert
                b_idx += 1
        scores = self.det.score_windows(batch)
        for s_i in range(m_steps):
            base = scores[s_i * (D + 1)]
            g = (scores[s_i * (D + 1) + 1 : (s_i + 1) * (D + 1)] - base) / eps
            total_grad += np.abs(g)
        ig = (total_grad / m_steps) * np.mean(np.abs(diff[-2:, :]), axis=0)
        attr = ig / (np.sum(ig) + 1e-12)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return attr, dt_ms

    def explain_lime(self, x_win: np.ndarray, n_samples: int = 240) -> Tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        W, D = x_win.shape
        Z = self.rng.binomial(1, 0.5, size=(n_samples, D)).astype(float)
        Z[0, :] = 1.0
        batch = np.zeros((n_samples, W, D))
        bg_draws = self.bg_windows[self.rng.integers(0, len(self.bg_windows), size=n_samples)]
        for i in range(n_samples):
            mask = Z[i][None, :]
            batch[i] = mask * x_win + (1.0 - mask) * bg_draws[i]
        y = self.det.score_windows(batch)
        dists = np.sqrt(np.mean((Z - 1.0) ** 2, axis=1))
        weights = np.exp(-(dists ** 2) / (0.25 ** 2))
        W_mat = Z * weights[:, None]
        reg = 1.0 * np.eye(D)
        beta = np.linalg.solve(W_mat.T @ Z + reg, W_mat.T @ (y - np.mean(y)))
        attr = np.maximum(beta, 0.0)
        if np.sum(attr) < 1e-9:
            attr = np.abs(beta)
        attr = attr / (np.sum(attr) + 1e-12)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return attr, dt_ms

    def explain_kernel_shap(self, x_win: np.ndarray, n_coalitions: int = 320) -> Tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        W, D = x_win.shape
        K_steps = 3
        P = K_steps * D
        Z = np.zeros((n_coalitions, P), dtype=float)
        weights = np.zeros(n_coalitions, dtype=float)
        for i in range(n_coalitions):
            s = self.rng.integers(1, P)
            chosen = self.rng.choice(P, size=s, replace=False)
            Z[i, chosen] = 1.0
            weights[i] = (P - 1.0) / (s * (P - s) + 1e-6)
        weights /= np.sum(weights)
        batch = np.repeat(x_win[None, :, :], n_coalitions + 2, axis=0)
        bg_draw = self.bg_windows[self.rng.integers(0, len(self.bg_windows), size=n_coalitions)]
        for i in range(n_coalitions):
            m_flat = Z[i].reshape(K_steps, D)
            batch[i, -K_steps:, :] = m_flat * x_win[-K_steps:, :] + (1.0 - m_flat) * bg_draw[i, -K_steps:, :]
        batch[-2] = self.bg_mean
        batch[-1] = x_win
        scores = self.det.score_windows(batch)
        v0 = scores[-2]
        y = scores[:-2] - v0
        W_mat = Z * weights[:, None]
        phi_flat = np.linalg.solve(W_mat.T @ Z + 0.15 * np.eye(P), W_mat.T @ y)
        phi_feat = np.sum(np.maximum(phi_flat.reshape(K_steps, D), 0.0), axis=0)
        if np.sum(phi_feat) < 1e-9:
            phi_feat = np.sum(np.abs(phi_flat.reshape(K_steps, D)), axis=0)
        attr = phi_feat / (np.sum(phi_feat) + 1e-12)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return attr, dt_ms

    def explain_shats(self, x_win: np.ndarray, m_factor: int = 12) -> Tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        W, D = x_win.shape
        n_coalitions = m_factor * D
        Z = np.zeros((n_coalitions, D), dtype=float)
        weights = np.zeros(n_coalitions, dtype=float)
        for i in range(n_coalitions):
            s = self.rng.integers(1, D)
            chosen = self.rng.choice(D, size=s, replace=False)
            Z[i, chosen] = 1.0
            weights[i] = (D - 1.0) / (s * (D - s) + 1e-6)
        weights /= np.sum(weights)
        batch = np.zeros((n_coalitions + 1, W, D))
        bg_draw = self.bg_windows[self.rng.integers(0, len(self.bg_windows), size=n_coalitions)]
        for i in range(n_coalitions):
            mask = Z[i][None, :]
            batch[i] = mask * x_win + (1.0 - mask) * bg_draw[i]
        batch[-1] = self.bg_mean
        scores = self.det.score_windows(batch)
        y = scores[:-1] - scores[-1]
        W_mat = Z * weights[:, None]
        phi = np.linalg.solve(W_mat.T @ Z + 0.08 * np.eye(D), W_mat.T @ y)
        attr = np.maximum(phi, 0.0)
        if np.sum(attr) < 1e-9:
            attr = np.abs(phi)
        attr = attr / (np.sum(attr) + 1e-12)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return attr, dt_ms

    def explain_pi_cax(self, x_win: np.ndarray, t_crit_s: float = 15.0,
                       budget_ms: float = 10.0,
                       use_causal_decoupling: bool = True,
                       use_manifold_shapley: bool = True,
                       use_phys_envelope: bool = True,
                       use_tier2_refinement: bool = True) -> Tuple[np.ndarray, float]:
        """
        Proposed PI-CAX: Physics-Informed Causal-Structural & Deadline-Aware Adaptive Explainability.
        """
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

        if use_causal_decoupling:
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
        else:
            z_causal = z_det.copy()

        if use_manifold_shapley:
            # Closed-Form Manifold Precision & Stage-Coalition Shapley Attribution:
            # 1. Precision-weighted quadratic form + directional persistence
            Lambda = self.scm_prec
            q_vec = np.abs(z_causal * (Lambda @ z_causal))
            diag_vec = (z_causal ** 2) * np.diag(Lambda)
            phi_tier1 = 0.60 * diag_vec + 0.40 * q_vec

            signed_det_persistence = np.abs(np.mean((r_all[-6:] - self.det.res_mean) / self.det.res_std, axis=0))
            coherence_factor = np.clip(signed_det_persistence / (np.mean(np.abs((r_all[-6:] - self.det.res_mean) / self.det.res_std), axis=0) + 1e-5), 0.10, 1.35)
            phi_tier1 = phi_tier1 * (0.20 + 0.95 * coherence_factor)

            # 2. Hierarchical Stage-Coalition Shapley Gating:
            # A genuine cyber-physical attack excites multiple coupled sensors/actuators within the attacked stage,
            # whereas an isolated cross-stage disturbance or adversarial decoy affects only 1-2 channels without stage-wide causal coupling!
            stage_active_counts = {}
            stage_total_energy = {}
            for s_id, inds in self.stage_map.items():
                active_in_stage = np.sum(z_det[inds] > 1.45) / float(len(inds))
                stage_active_counts[s_id] = active_in_stage
                stage_total_energy[s_id] = float(np.mean(phi_tier1[inds])) * (0.40 + 0.90 * active_in_stage)
            max_st_e = max(stage_total_energy.values()) + 1e-9
            for s_id, inds in self.stage_map.items():
                st_weight = (stage_total_energy[s_id] / max_st_e) ** 0.75
                phi_tier1[inds] *= (0.35 + 0.65 * st_weight)
        else:
            phi_tier1 = z_causal ** 2

        if use_phys_envelope:
            x_mean4 = np.mean(x_win[-4:], axis=0)
            low_viol = np.maximum(0.0, self.safety_bounds[:, 0] - x_mean4)
            high_viol = np.maximum(0.0, x_mean4 - self.safety_bounds[:, 1])
            bound_viol = low_viol + high_viol
            net_drift = np.abs(x_win[-1] - x_win[0]) / (np.sum(np.abs(np.diff(x_win, axis=0)), axis=0) + 1e-5)
            phys_mult = (1.0 + 1.15 * np.tanh(bound_viol)) * (0.50 + 0.95 * net_drift)
            phi_tier1 = phi_tier1 * phys_mult

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        effective_budget_ms = min(budget_ms, max(1.2, 0.45 * t_crit_s))
        if use_tier2_refinement and (elapsed_ms + 0.8 < effective_budget_ms):
            K_cand = min(5, D)
            top_cands = np.argsort(phi_tier1)[::-1][:K_cand]
            cf_batch = np.repeat(x_win[None, :, :], K_cand + 1, axis=0)
            cf_struct_gain = np.zeros(K_cand)
            for idx_c, f_c in enumerate(top_cands, start=1):
                for step_back in [-3, -2, -1]:
                    cf_batch[idx_c, step_back, f_c] = self.A_scm[f_c] @ cf_batch[idx_c, step_back - 1]
                cf_struct_gain[idx_c - 1] = signed_mean_shift[f_c] * (0.35 + 0.65 * oscillation_ratio[f_c])
            cf_scores = self.det.score_windows(cf_batch)
            base_score = cf_scores[0]
            cf_drops = np.maximum(0.0, base_score - cf_scores[1:])
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
