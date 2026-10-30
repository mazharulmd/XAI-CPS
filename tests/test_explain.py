import itertools
from math import factorial
import numpy as np
import torch
from xaicps import explain


def brute_shapley(v, L):
    phi = np.zeros(L)
    for i in range(L):
        others = [j for j in range(L) if j != i]
        for r in range(L):
            for S in itertools.combinations(others, r):
                m = sum(1 << j for j in S)
                w = factorial(r) * factorial(L - r - 1) / factorial(L)
                phi[i] += w * (v[m | (1 << i)] - v[m])
    return phi


def test_exact_shapley_matches_bruteforce_and_efficiency():
    rng = np.random.default_rng(0)
    for L in (3, 5, 7):
        v = rng.normal(size=1 << L); v[0] = 0
        phi = explain.exact_shapley(v, L)
        assert np.allclose(phi, brute_shapley(v, L))
        assert np.isclose(phi.sum(), v[-1] - v[0])


def test_symmetry_and_null_player():
    L = 4
    member, _, _ = explain._shapley_tables(L)
    # players 0 and 1 symmetric, player 3 null: v(S) = (|S cap {0,1}|)^2 + |S cap {2}|
    v = np.array([(m[0] + m[1]) ** 2 + 3 * m[2] for m in member], float)
    phi = explain.exact_shapley(v, L)
    assert np.isclose(phi[0], phi[1]) and np.isclose(phi[3], 0) and np.isclose(phi[2], 3)


class AdditiveDet:
    """e(Z) = sum_j a_j * Z[-1, j]  -> additive game in the tags."""
    def __init__(self, d, w=4):
        self.w = w; self.a = torch.arange(1, d + 1, dtype=torch.float32)
        self.model = None; self.sigma = torch.ones(d)
    def segment_score(self, Z):
        return (Z[:, -1] * self.a).sum(-1)


def test_group_exact_and_kernelshap_on_additive_game():
    d, w = 6, 4
    det = AdditiveDet(d, w)
    X = np.zeros((20, d), np.float32); X[15:] = 1.0      # tags jump from 0 (ref) to 1
    groups = [np.array([0, 1]), np.array([2]), np.array([3, 4, 5])]
    ctx = explain.Context(det, X, groups, t_ref=10)
    out = explain.lx(ctx, 16)
    assert np.allclose(out["signed"], [1 + 2, 3, 4 + 5 + 6], atol=1e-5)
    phi, n = explain.kernel_shap_players(ctx, 16, [np.array([j]) for j in range(d)], 256, None)
    assert np.allclose(phi, np.arange(1, d + 1), atol=1e-3)


def test_group_masks_cover_all_coalitions():
    groups = [np.array([0]), np.array([1, 2])]
    M = explain.group_masks(groups, 3)
    assert M.tolist() == [[False, False, False], [True, False, False], [False, True, True], [True, True, True]]


def test_tea_accumulates_and_confidence_bounded():
    seq = np.array([[1, 0, 0], [0, 1, 0], [0, 1, 0]], float)
    A, c = explain.tea(seq, np.array([2.0, 2.0, 2.0]), 1.0)
    assert A[-1].argmax() == 1 and np.all((c >= 0) & (c <= 1))
