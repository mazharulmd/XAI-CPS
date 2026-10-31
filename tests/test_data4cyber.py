import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from xai_cps.data4cyber import fit_normal_median_imputer, load_data4cyber_scenario
from scripts.download_data4cyber import safe_member_path
from xai_cps.detectors import RobustZDetector
from xai_cps.explainers import channel_groups, component_group_ablation


class Data4CyberLoaderTests(unittest.TestCase):
    def _write_pair(self, directory: Path, attack_labels=(True, True)) -> None:
        common = {
            "timestamp": ["2026-03-06T10:00:00Z", "2026-03-06T10:00:01Z"],
            "attack_phase": ["impact", "impact"],
            "attack_phase_all": ["impact", "impact"],
            "PV-Inverter-AC-Meter.freq": [50.0, 50.1],
            "PV-Inverter-AC-Meter.realtime": [1000, 1001],
            "Profile.timestamp": ["2025-09-20T09:43:00Z", "2025-09-20T09:43:01Z"],
        }
        attack = dict(common, attack_active=list(attack_labels))
        benign = dict(common, attack_active=[False, False], attack_phase=["normal"] * 2,
                      attack_phase_all=["normal"] * 2)
        pd.DataFrame(attack).to_csv(directory / "dataset_attack.csv", index=False)
        pd.DataFrame(benign).to_csv(directory / "dataset_benign.csv", index=False)

    def test_loader_excludes_all_label_and_clock_columns(self):
        with tempfile.TemporaryDirectory() as temp:
            scenario_dir = Path(temp) / "S1_example"
            scenario_dir.mkdir()
            self._write_pair(scenario_dir)
            attack, benign = load_data4cyber_scenario(scenario_dir)
            self.assertEqual(list(attack.features.columns), ["PV-Inverter-AC-Meter.freq"])
            self.assertEqual(attack.attack.tolist(), [True, True])
            self.assertEqual(benign.attack.tolist(), [False, False])
            self.assertEqual(attack.phase.tolist(), ["impact", "impact"])

    def test_loader_rejects_split_name_label_disagreement(self):
        with tempfile.TemporaryDirectory() as temp:
            scenario_dir = Path(temp) / "S1_example"
            scenario_dir.mkdir()
            self._write_pair(scenario_dir, attack_labels=(True, False))
            with self.assertRaisesRegex(ValueError, "disagrees"):
                load_data4cyber_scenario(scenario_dir)

    def test_imputer_uses_training_normal_statistics_only(self):
        train = pd.DataFrame({"sensor": [1.0, None, 3.0], "all_missing": [None, None, None]})
        imputer = fit_normal_median_imputer(train)
        test = pd.DataFrame({"sensor": [None, 1000000.0], "all_missing": [4.0, 5.0]})
        transformed = imputer.transform(test)
        self.assertEqual(imputer.feature_names, ["sensor"])
        self.assertEqual(transformed[:, 0].tolist(), [2.0, 1000000.0])

    def test_archive_member_path_rejects_traversal_and_accepts_normal_member(self):
        self.assertEqual(safe_member_path("data4cyber_dataset/S1/dataset.csv").as_posix(),
                         "data4cyber_dataset/S1/dataset.csv")
        for unsafe in ("../outside.txt", "/absolute.txt", "C:/outside.txt", "S1\\dataset.csv"):
            with self.subTest(unsafe=unsafe), self.assertRaises(ValueError):
                safe_member_path(unsafe)

    def test_data4cyber_component_groups_and_ablation(self):
        names = ["PV-Inverter-AC-Meter.freq", "PV-Inverter-AC-Meter.p_sum3", "Load-A-AC-Meter.freq"]
        self.assertEqual(channel_groups(names), ["PV-Inverter-AC-Meter", "PV-Inverter-AC-Meter", "Load-A-AC-Meter"])
        train_x = pd.DataFrame([[50.0, 1.0, 50.0], [50.1, 1.1, 50.1], [49.9, 0.9, 49.9]]).to_numpy()
        train_dx = train_x - train_x.mean(axis=0)
        model = RobustZDetector().fit(train_x, train_dx)
        scores = component_group_ablation(model, np.array([70.0, 20.0, 50.0]),
                                          np.array([20.0, 19.0, 0.0]), names)
        self.assertEqual(set(scores), {"PV-Inverter-AC-Meter", "Load-A-AC-Meter"})


if __name__ == "__main__":
    unittest.main()
