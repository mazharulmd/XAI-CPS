"""Statistical tests over attacks (the unit of analysis)."""
from __future__ import annotations
import numpy as np
from scipy import stats


def bootstrap_ci(x, n=10_000, alpha=0.05, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, (len(p) - k) * p[i])
        adj[i] = min(1.0, running)
    return adj


def rank_biserial(x, y):
    """Matched-pairs rank-biserial correlation (positive => x > y)."""
    d = np.asarray(x, float) - np.asarray(y, float)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def compare_to_reference(table, ref, higher_is_better=True):
    """table: dict method -> per-attack values (aligned). Wilcoxon signed-rank of `ref` vs
    every other method, Holm-corrected, with rank-biserial effect sizes; Friedman omnibus."""
    methods = [m for m in table if m != ref]
    M = np.array([table[m] for m in [ref] + methods], float)
    ok = np.all(np.isfinite(M), 0)
    M = M[:, ok]
    out = {"n": int(ok.sum())}
    try:
        out["friedman_chi2"], out["friedman_p"] = map(float, stats.friedmanchisquare(*M))
    except ValueError:
        out["friedman_chi2"], out["friedman_p"] = np.nan, np.nan
    raw, rows = [], []
    for i, m in enumerate(methods, start=1):
        x, y = M[0], M[i]
        if np.allclose(x, y):
            p = 1.0
        else:
            p = float(stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided").pvalue)
        eff = rank_biserial(x, y) if higher_is_better else -rank_biserial(x, y)
        raw.append(p); rows.append(dict(method=m, ref_mean=float(x.mean()), other_mean=float(y.mean()), effect_rb=eff))
    for r, p, pa in zip(rows, raw, holm(raw)):
        r["p"], r["p_holm"] = p, float(pa)
    out["pairs"] = rows
    return out
