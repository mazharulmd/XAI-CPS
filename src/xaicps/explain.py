"""Explainers for forecasting-based ICS anomaly detectors.

All explainers attribute the same quantity: the anomaly score e(Z_t) of the segment
Z_t = X[t-w : t+1], relative to a reference segment R (by default the incident's
last-known-good segment X[t_ref-w : t_ref+1]). A coalition of tags S is evaluated as
e(where(S, Z_t, R)), i.e. tags outside S are restored to their reference trajectory.

Each explainer returns a dict with
    tag   : [d] non-negative tag relevance (or None for loop-only methods)
    loop  : [L] non-negative loop relevance (tag scores summed per loop if tag-level)
    evals : number of detector forward evaluations used (hardware-independent cost)
"""
from __future__ import annotations

from functools import lru_cache
from math import factorial
import numpy as np
import torch


# ----------------------------------------------------------------------------- utils
class Context:
    """Everything an explainer needs for one incident on one test segment."""

    def __init__(self, det, X, groups, actuator=None, t_ref=None, background=None):
        self.det = det
        self.X = torch.from_numpy(X) if isinstance(X, np.ndarray) else X
        self.w = det.w
        self.d = self.X.shape[1]
        self.groups = [np.asarray(g) for g in groups]           # loop -> tag indices
        self.L = len(self.groups)
        self.tag2loop = np.full(self.d, -1)
        for i, g in enumerate(self.groups):
            self.tag2loop[g] = i
        self.actuator = np.zeros(self.d, bool) if actuator is None else actuator
        self.t_ref = t_ref
        self.background = background                            # [K, w+1, d] tensor or None

    def seg(self, t):
        return self.X[t - self.w:t + 1]

    def ref(self):
        return self.seg(self.t_ref)

    def to_loops(self, tag_scores):
        out = np.zeros(self.L)
        np.add.at(out, self.tag2loop[self.tag2loop >= 0], tag_scores[self.tag2loop >= 0])
        return out


def _eval_masks(ctx, cur, ref, masks, chunk=2048):
    """masks: bool [C, d] (True = take current value). Returns e [C] (numpy)."""
    out = []
    with torch.no_grad():
        for b in range(0, len(masks), chunk):
            m = torch.from_numpy(masks[b:b + chunk])[:, None, :]
            Z = torch.where(m, cur[None], ref[None])
            out.append(ctx.det.segment_score(Z).numpy())
    return np.concatenate(out).astype(np.float64)


def _eval_masks_bg(ctx, cur, masks, chunk=2048):
    """Interventional value with a background set: mean_k e(where(S, cur, bg_k))."""
    bg = ctx.background
    K = len(bg)
    out = np.zeros(len(masks))
    with torch.no_grad():
        for b in range(0, len(masks), max(1, chunk // K)):
            m = torch.from_numpy(masks[b:b + max(1, chunk // K)])[:, None, None, :]   # [c,1,1,d]
            Z = torch.where(m, cur[None, None], bg[None])                              # [c,K,w+1,d]
            e = ctx.det.segment_score(Z.reshape(-1, *cur.shape)).reshape(len(m), K)
            out[b:b + len(m)] = e.mean(1).numpy()
    return out


@lru_cache(maxsize=None)
def _shapley_tables(L):
    """All 2^L coalitions as bitmasks, membership matrix and Shapley weights per size."""
    n = 1 << L
    member = ((np.arange(n)[:, None] >> np.arange(L)[None]) & 1).astype(bool)   # [n, L]
    size = member.sum(1)
    wsize = np.array([factorial(s) * factorial(L - s - 1) / factorial(L) for s in range(L)])
    return member, size, wsize


def exact_shapley(v, L):
    """v: [2^L] coalition values indexed by bitmask. Returns phi [L]."""
    member, size, wsize = _shapley_tables(L)
    phi = np.zeros(L)
    for i in range(L):
        S = np.flatnonzero(~member[:, i])
        phi[i] = np.sum(wsize[size[S]] * (v[S | (1 << i)] - v[S]))
    return phi


def group_masks(groups, d):
    """Tag masks [2^G, d] for every coalition of the given tag groups."""
    G = len(groups)
    member, _, _ = _shapley_tables(G)
    M = np.zeros((1 << G, d), bool)
    for gi, g in enumerate(groups):
        M[np.ix_(member[:, gi], g)] = True
    return M


# ----------------------------------------------------------------------------- group-exact
def group_exact(ctx, t, groups=None, use_background=False, base_mask=None):
    """Exact Shapley values over tag groups (default: the control loops).

    base_mask: tags outside all groups that are always kept at their current value."""
    groups = ctx.groups if groups is None else groups
    cur = ctx.seg(t)
    M = group_masks(groups, ctx.d)
    if base_mask is not None:
        M |= base_mask[None]
    v = _eval_masks_bg(ctx, cur, M) if use_background else _eval_masks(ctx, cur, ctx.ref(), M)
    phi = exact_shapley(v - v[0], len(groups))
    evals = len(M) * (len(ctx.background) if use_background else 1)
    return phi, v, evals


def lx(ctx, t):
    """LX: loop-exact Shapley with the last-known-good reference (proposed)."""
    phi, v, n = group_exact(ctx, t)
    return dict(tag=None, loop=np.maximum(phi, 0), signed=phi, evals=n, total=v[-1] - v[0])


def lx_background(ctx, t):
    """Loop-exact Shapley with a training background (ShaTS-style multi-feature grouping)."""
    phi, v, n = group_exact(ctx, t, use_background=True)
    return dict(tag=None, loop=np.maximum(phi, 0), signed=phi, evals=n)


def partition_exact(ctx, t, partition):
    """Exact Shapley over an arbitrary tag partition; mapped to loops by spreading each
    group's value uniformly over its tags (used for cluster / random-partition controls)."""
    phi, v, n = group_exact(ctx, t, groups=partition)
    tag = np.zeros(ctx.d)
    for g, p in zip(partition, np.maximum(phi, 0)):
        tag[g] = p / len(g)
    return dict(tag=tag, loop=ctx.to_loops(tag), evals=n)


def refine_tags(ctx, t, loop_idx, max_exact=12, budget=4096, rng=None):
    """H-LX: tag attribution inside one loop (other loops kept at current values)."""
    g = ctx.groups[loop_idx]
    base = np.ones(ctx.d, bool); base[g] = False
    if len(g) <= max_exact:
        phi, _, n = group_exact(ctx, t, groups=[np.array([j]) for j in g], base_mask=base)
    else:
        phi, n = kernel_shap_players(ctx, t, [np.array([j]) for j in g], budget, rng, base_mask=base)
    out = np.zeros(ctx.d); out[g] = np.maximum(phi, 0)
    return out, n


# ----------------------------------------------------------------------------- tag-level baselines
def residual_error(ctx, t):
    """RE: squared normalised residual per tag (Fung et al.'s 'MSE' ranking)."""
    with torch.no_grad():
        Z = ctx.seg(t)[None]
        pred = ctx.det.model(Z[:, :-1])
        from .detectors import soft_clip
        r = soft_clip((Z[:, -1] - pred) / ctx.det.sigma)[0].numpy()
    tag = r.astype(np.float64) ** 2
    return dict(tag=tag, loop=ctx.to_loops(tag), evals=1)


def _grad(ctx, Z):
    Z = Z.clone().requires_grad_(True)
    e = ctx.det.segment_score(Z).sum()
    (g,) = torch.autograd.grad(e, Z)
    return g


def saliency(ctx, t):
    """SM: |de/dZ| summed over the segment."""
    g = _grad(ctx, ctx.seg(t)[None])[0]
    tag = g.abs().sum(0).numpy().astype(np.float64)
    return dict(tag=tag, loop=ctx.to_loops(tag), evals=1)


def integrated_gradients(ctx, t, steps=32):
    """IG from the reference segment to the current one (positive part)."""
    cur, ref = ctx.seg(t), ctx.ref()
    a = torch.linspace(0, 1, steps + 1)[1:, None, None]
    Z = ref[None] + a * (cur - ref)[None]
    g = _grad(ctx, Z).mean(0)
    tag = np.maximum(((cur - ref) * g).sum(0).numpy().astype(np.float64), 0)
    return dict(tag=tag, loop=ctx.to_loops(tag), evals=steps)


def kernel_shap_players(ctx, t, players, budget, rng, base_mask=None):
    """KernelSHAP over arbitrary player groups with a coalition budget."""
    rng = np.random.default_rng(0) if rng is None else rng
    M = len(players)
    sizes = np.arange(1, M)
    p = (M - 1) / (sizes * (M - sizes)); p /= p.sum()
    nb = max(budget - 2, M + 1)
    Zb = np.zeros((nb, M), bool)
    for k, s in enumerate(rng.choice(sizes, nb, p=p)):
        Zb[k, rng.choice(M, s, replace=False)] = True
    Zall = np.vstack([np.zeros((1, M), bool), np.ones((1, M), bool), Zb])
    masks = np.zeros((len(Zall), ctx.d), bool)
    for j, g in enumerate(players):
        masks[:, g] |= Zall[:, j:j + 1]
    if base_mask is not None:
        masks |= base_mask[None]
    v = _eval_masks(ctx, ctx.seg(t), ctx.ref(), masks)
    v0, v1 = v[0], v[1]
    y = v[2:] - v0 - Zb[:, -1] * (v1 - v0)
    A = Zb[:, :-1].astype(float) - Zb[:, -1:].astype(float)
    phi = np.linalg.solve(A.T @ A + 1e-6 * np.eye(M - 1), A.T @ y)
    phi = np.append(phi, (v1 - v0) - phi.sum())
    return phi, len(masks)


def kernel_shap(ctx, t, budget=512, rng=None):
    phi, n = kernel_shap_players(ctx, t, [np.array([j]) for j in range(ctx.d)], budget, rng)
    tag = np.maximum(phi, 0)
    return dict(tag=tag, loop=ctx.to_loops(tag), evals=n)


def lime(ctx, t, budget=512, rng=None):
    """LIME with a binary interpretable representation (tag kept vs restored)."""
    rng = np.random.default_rng(0) if rng is None else rng
    d = ctx.d
    Zb = rng.random((budget - 1, d)) < 0.5
    Zb = np.vstack([np.ones((1, d), bool), Zb])
    v = _eval_masks(ctx, ctx.seg(t), ctx.ref(), Zb)
    dist = np.sqrt(d - Zb.sum(1))
    wts = np.exp(-dist ** 2 / (0.75 * np.sqrt(d)) ** 2)
    A = np.hstack([np.ones((len(Zb), 1)), Zb.astype(float)])
    W = wts[:, None]
    coef = np.linalg.solve(A.T @ (W * A) + np.diag([0] + [1.0] * d), A.T @ (wts * v))[1:]
    tag = np.maximum(coef, 0)
    return dict(tag=tag, loop=ctx.to_loops(tag), evals=budget)


def shats_sensor(ctx, t, m_per_player=None, rng=None):
    """ShaTS (feature grouping, approximate): stratified sampling of m coalitions per
    player (default m = 20 * |G| as in Franco de la Pena et al.), two evaluations each."""
    rng = np.random.default_rng(0) if rng is None else rng
    d = ctx.d
    m = 20 * d if m_per_player is None else m_per_player
    per_size = max(1, m // d)
    cur, ref = ctx.seg(t), ctx.ref()
    phi = np.zeros(d); evals = 0
    for i in range(d):
        others = np.delete(np.arange(d), i)
        with_i, without_i = [], []
        for s in range(d):
            for _ in range(per_size):
                S = rng.choice(others, s, replace=False) if s > 0 else np.array([], int)
                mk = np.zeros(d, bool); mk[S] = True
                without_i.append(mk.copy()); mk[i] = True; with_i.append(mk)
        Mw, Mo = np.array(with_i), np.array(without_i)
        dv = _eval_masks(ctx, cur, ref, Mw) - _eval_masks(ctx, cur, ref, Mo)
        phi[i] = dv.reshape(d, per_size).mean(1).mean()
        evals += 2 * len(Mw)
    tag = np.maximum(phi, 0)
    return dict(tag=tag, loop=ctx.to_loops(tag), evals=evals)


def _norm(x):
    s = x.sum()
    return x / s if s > 0 else x


def fung_ensemble(ctx, t, parts=None, beta=2.5, budget=512, rng=None):
    """Fung et al. (NDSS'24) ensemble: normalised RE + SM + (LEMNA -> LIME here);
    beta-weighted ML attributions for actuator tags."""
    parts = parts or {}
    re = parts.get("RE") or residual_error(ctx, t)
    sm = parts.get("SM") or saliency(ctx, t)
    li = parts.get(f"LIME") or lime(ctx, t, budget, rng)
    a = ctx.actuator
    tag = _norm(re["tag"]) + np.where(a, beta, 1.0) * (_norm(sm["tag"]) + _norm(li["tag"]))
    return dict(tag=tag, loop=ctx.to_loops(tag), evals=re["evals"] + sm["evals"] + li["evals"])


# ----------------------------------------------------------------------------- partitions
def correlation_clusters(train, n_clusters, seed=0):
    """Data-driven tag partition (average-linkage on 1-|corr|), like Partition/Owen SHAP."""
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    X = np.concatenate(train, 0)[::10]
    C = np.nan_to_num(np.corrcoef(X.T))
    D = 1 - np.abs(C); np.fill_diagonal(D, 0)
    Zl = linkage(squareform(D, checks=False), "average")
    lab = fcluster(Zl, n_clusters, "maxclust")
    return [np.flatnonzero(lab == k) for k in np.unique(lab)]


def random_partition(groups, d, seed=0):
    """Random partition with the same group sizes as the loops (structure control)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(np.concatenate(groups))
    out, k = [], 0
    for g in groups:
        out.append(np.sort(perm[k:k + len(g)])); k += len(g)
    return out


# ----------------------------------------------------------------------------- temporal evidence accumulation
def tea(loop_seq, scores, theta, weighting="logexcess", eps=1e-3):
    """Temporal evidence accumulation over a sequence of loop attributions.

    loop_seq: [K, L] non-negative attributions at successive explanation times;
    scores: [K] detector EWMA scores at those times. Returns cumulative evidence [K, L]
    and confidence margins [K]."""
    P = np.asarray(loop_seq, float)
    P = P / np.maximum(P.sum(1, keepdims=True), 1e-12)
    if weighting == "uniform":
        w = np.ones(len(P))
    else:
        w = np.maximum(np.log(np.maximum(np.asarray(scores), 1e-12) / theta), 0) + eps
    A = np.cumsum(w[:, None] * P, 0)
    srt = np.sort(A, 1)
    conf = (srt[:, -1] - srt[:, -2]) / np.maximum(A.sum(1), 1e-12)
    return A, conf
