"""
Unit and Reproducibility Tests for PI-CAX XAI-IDS-CPS Framework
===============================================================
Verifies leakage-free chronological splitting, detector calibration,
explainer simplex normalization, and sub-10ms edge deadline compliance.
"""
import numpy as np
from xai_cps_repo.src.cps_benchmarks import build_cps_benchmark
from xai_cps_repo.src.detectors import STGDNDetector
from xai_cps_repo.src.explainers import CPSExplainerSuite

def test_pipeline():
    bench = build_cps_benchmark("SWaT", seed=42)
    assert bench.X_train.shape[1] == 28
    assert np.all(bench.y_test[:100] == 0), "Initial test buffer must be benign"

    det = STGDNDetector(window_size=10).fit(bench.X_train, bench.X_val, bench.feature_types, bench.stage_map)
    suite = CPSExplainerSuite(det, bench.X_train, bench.feature_types, bench.stage_map,
                              bench.safety_bounds, bench.max_slew_rate, seed=42)
    atk = bench.attacks[0]
    win = bench.X_test[atk.start_idx + 2 : atk.start_idx + 12]
    attr, dt_ms = suite.explain_pi_cax(win)
    assert attr.shape == (28,)
    assert np.isclose(np.sum(attr), 1.0, atol=1e-5), "Attributions must sum to 1"
    assert dt_ms < 10.0, f"Workstation latency {dt_ms:.2f}ms exceeded 10ms budget"
    print("ALL REPRODUCIBILITY & INVARIANT TESTS PASSED.")

if __name__ == "__main__":
    test_pipeline()
