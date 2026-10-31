import unittest

from scripts.evaluate_hai_target_localization import random_ranking_benchmark_tests


def _run(seed: int, hits: dict[str, dict[str, float]]) -> dict:
    events = []
    for attack_id, hit_values in hits.items():
        events.append({
            "attack_id": attack_id,
            "dataset_file": f"{attack_id}.csv",
            "random_reference_hit_probability_at_1": 0.2 if attack_id == "a" else 0.8,
            "random_reference_hit_probability_at_3": 0.2 if attack_id == "a" else 0.8,
            "random_reference_hit_probability_at_5": 0.2 if attack_id == "a" else 0.8,
            "native_hit_at_1": hit_values["native"],
            "native_hit_at_3": hit_values["native"],
            "native_hit_at_5": hit_values["native"],
            "grouped_ablation_hit_at_1": hit_values["grouped_ablation"],
            "grouped_ablation_hit_at_3": hit_values["grouped_ablation"],
            "grouped_ablation_hit_at_5": hit_values["grouped_ablation"],
        })
    return {"seed": seed, "event_outcomes": {"PCA-TEMPORAL": events}}


class HaiRandomRankingTests(unittest.TestCase):
    def test_random_reference_is_paired_by_attack_and_seed(self):
        runs = [
            _run(1, {"a": {"native": 1.0, "grouped_ablation": 1.0},
                     "b": {"native": 0.0, "grouped_ablation": 0.0}}),
            _run(2, {"a": {"native": 1.0, "grouped_ablation": 1.0},
                     "b": {"native": 0.0, "grouped_ablation": 0.0}}),
        ]
        result = random_ranking_benchmark_tests(runs, seed=17, draws=1000)
        native = result["PCA-TEMPORAL"]["native"]["5"]
        self.assertAlmostEqual(native["observed_unconditional_hit_rate"], 0.5)
        self.assertAlmostEqual(native["random_reference_unconditional_hit_rate"], 0.5)
        self.assertAlmostEqual(native["excess_hit_rate_over_random_reference"], 0.0)
        self.assertEqual(native["n_attack_event_clusters"], 2)
        self.assertEqual(native["n_test_file_clusters"], 2)
        self.assertEqual(len(native["event_cluster_bootstrap_ci95_for_excess"]), 2)
        self.assertEqual(len(native["file_cluster_bootstrap_ci95_for_excess"]), 2)

    def test_randomization_draw_count_is_guarded(self):
        with self.assertRaises(ValueError):
            random_ranking_benchmark_tests([], draws=999)


if __name__ == "__main__":
    unittest.main()
