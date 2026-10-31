import numpy as np
import pytest

pytest.importorskip("sklearn")

from xai_cps.isolation_forest import HAIIsolationForest


def test_normal_only_isolation_forest_scores_finite_and_exposes_model_size():
    rng = np.random.default_rng(31)
    x = rng.normal(size=(300, 4))
    dx = np.diff(x, axis=0, prepend=x[[0]])
    model = HAIIsolationForest(n_estimators=25, max_samples=64, random_state=13, n_jobs=1).fit(x, dx)
    score = model.score(x[:15], dx[:15])
    assert score.shape == (15,)
    assert np.isfinite(score).all()
    assert model.model_bytes() > 0
    z, zd = model.standardized(x[:2], dx[:2])
    assert z.shape == zd.shape == (2, 4)


def test_isolation_forest_rejects_malformed_and_nonfinite_data():
    x = np.ones((30, 2))
    dx = np.zeros_like(x)
    with pytest.raises(ValueError, match="same-shaped"):
        HAIIsolationForest(n_estimators=5, max_samples=8).fit(x, dx[:, :1])
    model = HAIIsolationForest(n_estimators=5, max_samples=8, n_jobs=1).fit(x, dx)
    bad = x[:2].copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        model.score(bad, dx[:2])

