import numpy as np
import pytest

from xai_cps.geco_reference import GeCoAR1Reference


def test_geco_ar1_scores_are_finite_and_reset_at_sequence_start():
    rng = np.random.default_rng(7)
    x = np.zeros((240, 3), dtype=float)
    for i in range(1, len(x)):
        x[i] = 0.85 * x[i - 1] + rng.normal(0.0, 0.03, size=3)
    model = GeCoAR1Reference().fit([x[:120], x[120:]])
    channels, overall, latency_ms = model.score_sequence(x[180:])
    assert channels.shape == (60, 3)
    assert overall.shape == (60,)
    assert latency_ms.shape == (59,)
    assert np.isfinite(channels).all()
    assert np.isfinite(overall).all()
    assert np.isfinite(latency_ms).all()
    assert np.all(channels[0] == 0.0)
    assert overall[0] == 0.0


def test_geco_ar1_rejects_nonfinite_inputs_and_schema_mismatch():
    model = GeCoAR1Reference().fit([np.arange(24, dtype=float).reshape(12, 2)])
    with pytest.raises(ValueError, match="feature count"):
        model.score_sequence(np.ones((5, 3)))
    bad = np.ones((5, 2))
    bad[2, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        model.score_sequence(bad)


def test_geco_ar1_requires_separate_fit_and_residual_data():
    with pytest.raises(ValueError, match="transitions"):
        GeCoAR1Reference().fit([np.arange(6, dtype=float).reshape(3, 2)])

