import numpy as np
import torch
from xaicps import explain
from xaicps.detectors import Detector
from xaicps.picax import PICAX


def _toy(seed=0, d=6, n=3000):
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(scale=0.05, size=(n, d)), 0).astype(np.float32)
    return x - x.mean(0)


def test_picax_is_deterministic_normalised_and_finds_persistent_shift():
    train = [_toy(0)]
    det = Detector("linear", 6, w=8, seed=0, alpha=0.1, q=0.999)
    det.fit(train, [_toy(1)], max_windows=2000)
    groups = [np.array([0, 1]), np.array([2, 3]), np.array([4, 5])]
    pc = PICAX(det, train, groups, np.array([False, True] * 3))
    X = _toy(2)
    X[1500:, 2] += 3.0                                 # persistent one-sided shift in loop 1
    ctx = explain.Context(det, X, groups)
    a, b = pc.explain(ctx, 1520), pc.explain(ctx, 1520)
    assert np.allclose(a["tag"], b["tag"]) and np.isclose(a["tag"].sum(), 1.0) and (a["tag"] >= 0).all()
    assert a["evals"] == 16 and int(np.argmax(a["loop"])) == 1
    off = pc.explain(ctx, 1520, causal=False, manifold=False, envelope=False, tier2=False)
    assert np.isclose(off["tag"].sum(), 1.0)
