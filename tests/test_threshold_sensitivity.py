import unittest

import numpy as np

from xai_cps.threshold_sensitivity import aggregate_runs, validation_thresholds


class ThresholdSensitivityTests(unittest.TestCase):
    def test_quantiles_are_calibrated_from_supplied_scores(self):
        got = validation_thresholds(np.array([0.0, 1.0, 2.0, 3.0]), [0.5, 0.75])
        self.assertEqual(got["0.500000"], 1.5)
        self.assertEqual(got["0.750000"], 2.25)

    def test_invalid_quantile_rejected(self):
        with self.assertRaises(ValueError):
            validation_thresholds(np.array([1.0]), [1.0])


    def test_repeat_aggregation_reports_mean_and_range_not_ci(self):
        runs = []
        for recall, fpr in [(0.8, 0.01), (1.0, 0.005)]:
            runs.append({"models": {"M": {"0.990000": {
                "threshold": 2.0, "validation_fpr_observed": fpr,
                "pooled_point": {"precision": 0.5, "recall": recall, "f1": 0.6, "average_precision": 0.7, "false_positive_rate": fpr},
                "pooled_event": {"event_recall": recall, "mean_detection_delay_seconds_weighted": 3.0, "false_alarm_events_per_normal_hour": 2.0, "attack_events": 5}
            }}}})
        result = aggregate_runs(runs)["M"]["0.990000"]
        self.assertEqual(result["n_repeated_fits"], 2)
        self.assertEqual(result["event_recall"]["mean"], 0.9)
        self.assertEqual(result["event_recall"]["seed_range"], [0.8, 1.0])

    def test_nonfinite_validation_scores_rejected(self):
        with self.assertRaises(ValueError):
            validation_thresholds(np.array([1.0, np.nan]), [0.9])


if __name__ == "__main__":
    unittest.main()
