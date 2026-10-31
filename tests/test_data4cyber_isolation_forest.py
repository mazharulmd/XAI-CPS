import numpy as np
import pytest

pytest.importorskip("sklearn")

from xai_cps.data4cyber_isolation_forest import Data4CyberIsolationForest
from xai_cps.explainers import component_group_ablation


def test_data4cyber_if_scores_are_finite_and_higher_for_an_outlier():
    rng = np.random.default_rng(19)
    x = rng.normal(0.0, 0.1, size=(400, 3))
    dx = np.vstack([np.zeros((1, 3)), np.diff(x, axis=0)])
    model = Data4CyberIsolationForest(random_state=19, n_estimators=80, max_samples=128).fit(x, dx)
    normal = model.score(x[:20], dx[:20])
    outlier = model.score(np.array([[8.0, 0.0, 0.0]]), np.array([[8.0, 0.0, 0.0]]))
    assert np.isfinite(normal).all()
    assert outlier[0] > np.median(normal)


def test_group_ablation_uses_fitted_normal_references_and_stays_finite():
    rng = np.random.default_rng(20)
    x = rng.normal(size=(100, 3))
    dx = np.vstack([np.zeros((1, 3)), np.diff(x, axis=0)])
    model = Data4CyberIsolationForest(random_state=20, n_estimators=50, max_samples=64).fit(x, dx)
    values = component_group_ablation(
        model, x[5], dx[5], ["meter_a", "meter_b", "pump_a"], previous_x=x[4]
    )
    assert set(values) == {"meter", "pump"}
    assert np.isfinite(list(values.values())).all()

