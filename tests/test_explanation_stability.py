import unittest

import numpy as np

from scripts.evaluate_hai_explanation_stability import aggregate, topk_jaccard


class ExplanationStabilityTests(unittest.TestCase):
    def test_topk_jaccard(self):
        self.assertEqual(topk_jaccard(np.array([3, 2, 1]), np.array([3, 2, 1]), k=2), 1.0)
        self.assertEqual(topk_jaccard(np.array([3, 2, 1]), np.array([1, 2, 3]), k=1), 0.0)

    def test_factorial_comparisons_hold_baseline_explainer_fixed(self):
        rows = []
        for event in range(4):
            rows.append({
                "run_id": "seed42", "seed": 42, "model": "PCA-TEMPORAL",
                "file": "test1.csv.gz", "event_index": event, "state_only": False,
                "legacy_explainer_invalid_input_2pct_jaccard": 0.95,
                "legacy_explainer_valid_input_2pct_jaccard": 0.60,
                "identity_preserving_explainer_invalid_input_2pct_jaccard": 0.90,
                "identity_preserving_0.02_jaccard": 0.70,
                "identity_preserving_0.05_jaccard": 0.55, "perturbations": 5,
            })
        result = aggregate(rows, (0.02, 0.05), 200, 1)
        comparisons = result["paired_comparisons"]["PCA-TEMPORAL"]
        self.assertAlmostEqual(comparisons["legacy_explainer_invalid_minus_valid_input_2pct"]["mean_paired_event_difference"], 0.35)
        self.assertAlmostEqual(comparisons["identity_preserving_explainer_invalid_minus_valid_input_2pct"]["mean_paired_event_difference"], 0.20)
        self.assertEqual(comparisons["identity_preserving_explainer_invalid_minus_valid_input_2pct"]["n_matched_events"], 4)
        self.assertAlmostEqual(comparisons["difference_in_input_design_effect_by_ablation_semantics"]["mean_paired_event_difference"], 0.15)

    def test_state_only_model_excluded_from_temporal_input_comparisons(self):
        rows = [{
            "run_id": "seed42", "seed": 42, "model": "PCA-STATE",
            "file": "test1.csv.gz", "event_index": 0, "state_only": True,
            "legacy_explainer_invalid_input_2pct_jaccard": 1.0,
            "legacy_explainer_valid_input_2pct_jaccard": None,
            "identity_preserving_explainer_invalid_input_2pct_jaccard": None,
            "identity_preserving_0.02_jaccard": 1.0,
            "identity_preserving_0.05_jaccard": 1.0, "perturbations": 5,
        }]
        result = aggregate(rows, (0.02, 0.05), 50, 2)
        comparisons = result["paired_comparisons"]["PCA-STATE"]
        self.assertNotIn("legacy_explainer_invalid_minus_valid_input_2pct", comparisons)
        self.assertNotIn("identity_preserving_explainer_invalid_minus_valid_input_2pct", comparisons)


if __name__ == "__main__":
    unittest.main()
