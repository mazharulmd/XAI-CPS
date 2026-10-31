import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_hai_target_localization_by_action import action_class, hit, split_tokens


class TargetActionStrataTests(unittest.TestCase):
    def test_documented_action_types_compose_across_scenarios(self):
        mapping = pd.read_csv(ROOT / "data/annotations/hai21_scenario_action_types.csv")
        action_types = {row.scenario_id: split_tokens(row.variable_types) for row in mapping.itertuples(index=False)}
        self.assertEqual(action_class({"AP01"}, action_types), "SP")
        self.assertEqual(action_class({"AP03", "AP04"}, action_types), "CV+PV")
        self.assertEqual(action_class({"AP02", "AP18"}, action_types), "CV+PV+SP")

    def test_unknown_scenarios_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "No official action-type annotation"):
            action_class({"AP99"}, {"AP01": {"SP"}})

    def test_missed_or_empty_rankings_do_not_hit_targets(self):
        targets = {"P1_B2016"}
        self.assertEqual(hit(None, targets, 5), 0.0)
        self.assertEqual(hit(float("nan"), targets, 5), 0.0)
        self.assertEqual(hit("P1_B2016|P1_PIT01", targets, 1), 1.0)
        self.assertEqual(hit("P1_PIT01|P1_B2016", targets, 1), 0.0)


if __name__ == "__main__":
    unittest.main()
