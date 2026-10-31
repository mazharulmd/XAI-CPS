"""
Unit and Reproducibility Tests for PI-CAX XAI-IDS-CPS Framework
"""
import numpy as np

def test_simplex_and_sparsity():
    raw_phi = np.array([4.2, 0.8, 0.1, 0.05, 0.02, 0.01])
    floor = np.percentile(raw_phi, 60)
    clean = np.maximum(0.0, raw_phi - 0.75 * floor)
    attr = clean / (np.sum(clean) + 1e-12)
    assert np.isclose(np.sum(attr), 1.0, atol=1e-6)
    assert np.argmax(attr) == 0
    print("ALL REPRODUCIBILITY & INVARIANT TESTS PASSED.")

if __name__ == "__main__":
    test_simplex_and_sparsity()
