import unittest

from xai_cps.target_eval import (
    aggregate_component_scores,
    component_hit_at_k,
    pump_component_from_tag,
    rank_components,
)


class TargetEvaluationTests(unittest.TestCase):
    def test_pump_component_mapping_is_exact(self):
        self.assertEqual(pump_component_from_tag("F_PU2"), "PU2")
        self.assertEqual(pump_component_from_tag("S_PU10"), "PU10")
        self.assertIsNone(pump_component_from_tag("P_J280"))

    def test_component_aggregation_sums_nonnegative_tag_scores(self):
        scores = aggregate_component_scores({
            "F_PU10": 0.4,
            "S_PU10": 0.6,
            "F_PU2": 0.7,
            "P_J280": 99.0,
        })
        self.assertEqual(scores, {"PU10": 1.0, "PU2": 0.7})
        self.assertEqual(rank_components(scores), ["PU10", "PU2"])

    def test_component_hit_counts_any_documented_target(self):
        self.assertTrue(component_hit_at_k(["PU10", "PU2"], ["PU2"], 2))
        self.assertFalse(component_hit_at_k(["PU10", "PU2"], ["PU2"], 1))
        self.assertTrue(component_hit_at_k(["PU10"], ["PU10", "PU11"], 1))

    def test_component_hit_rejects_nonpositive_k(self):
        with self.assertRaises(ValueError):
            component_hit_at_k(["PU10"], ["PU10"], 0)


if __name__ == "__main__":
    unittest.main()
