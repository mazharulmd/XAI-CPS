from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from xai_cps.data import load_hai
from xai_cps.detectors import PCAResidualDetector, RobustZDetector
from xai_cps.metrics import binary_metrics, event_metrics
from xai_cps.threshold_sensitivity import aggregate_runs, validation_thresholds


def evaluate_run(run_dir: Path, config_path: Path, quantiles: list[float]) -> dict:
    run_dir = run_dir.resolve()
    run_summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    seed = int(run_summary["seed"])
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    dataset = load_hai(config_path, max_train_rows=int(config["training_rows_max"]), seed=seed)
    zeros_train = np.zeros_like(dataset.train_dx)
    zeros_val = np.zeros_like(dataset.validation.dx)
    models = [
        ("RobustZ-TEMPORAL", RobustZDetector(), dataset.train_dx, dataset.validation.dx),
        ("PCA-STATE", PCAResidualDetector(config["variance_explained"], config["max_components"]), zeros_train, zeros_val),
        ("PCA-TEMPORAL", PCAResidualDetector(config["variance_explained"], config["max_components"]), dataset.train_dx, dataset.validation.dx),
    ]
    predictions = pd.read_csv(run_dir / "predictions.csv.gz", compression="gzip")
    required = {"file", "timestamp", "attack", "score", "model"}
    if not required.issubset(predictions.columns):
        raise ValueError(f"prediction file missing columns: {sorted(required - set(predictions.columns))}")
    result = {"run_id": run_summary["run_id"], "run_dir": str(run_dir), "seed": seed, "models": {}}
    for model_name, detector, train_dx, val_dx in models:
        detector.fit(dataset.train_x, train_dx)
        val_scores = detector.score(dataset.validation.x, val_dx)
        thresholds = validation_thresholds(val_scores, quantiles)
        source_threshold = float(run_summary["models"][model_name]["threshold"])
        q99_key = f"{1.0 - float(config['validation_false_positive_rate']):.6f}"
        if q99_key in thresholds and not np.isclose(thresholds[q99_key], source_threshold, rtol=1e-8, atol=1e-8):
            raise RuntimeError(f"refitted {model_name} validation threshold does not reproduce run {run_summary['run_id']}")
        model_rows = predictions[predictions["model"] == model_name]
        by_quantile = {}
        for q_key, threshold in thresholds.items():
            file_rows = []
            ordered_groups = []
            for filename, frame in model_rows.groupby("file", sort=True):
                frame = frame.sort_values("timestamp", kind="mergesort")
                ordered_groups.append(frame)
                y = frame["attack"].to_numpy(dtype=bool)
                scores = frame["score"].to_numpy(dtype=float)
                pred = scores > threshold
                timestamps = pd.to_datetime(frame["timestamp"], errors="raise").to_numpy()
                file_rows.append({
                    "file": str(filename),
                    "point": binary_metrics(y, scores, threshold),
                    "events": event_metrics(y, pred, timestamps),
                })
            if not file_rows:
                raise RuntimeError(f"no saved predictions found for {model_name}")
            pooled_y = np.concatenate([g["attack"].to_numpy(dtype=bool) for g in ordered_groups])
            pooled_scores = np.concatenate([g["score"].to_numpy(dtype=float) for g in ordered_groups])
            pooled_point = binary_metrics(pooled_y, pooled_scores, threshold)
            total_events = sum(row["events"]["attack_events"] for row in file_rows)
            detected_events = sum(row["events"]["detected_attack_events"] for row in file_rows)
            false_events = sum(row["events"]["false_alarm_events"] for row in file_rows)
            normal_rows = sum(int(np.sum(~g["attack"].to_numpy(dtype=bool))) for g in ordered_groups)
            normal_hours = normal_rows / float(config.get("sampling_hz", 1)) / 3600.0
            delay_weighted = sum((row["events"]["mean_detection_delay_seconds"] or 0.0) * row["events"]["detected_attack_events"] for row in file_rows)
            by_quantile[q_key] = {
                "threshold": float(threshold),
                "validation_fpr_observed": float(np.mean(val_scores > threshold)),
                "pooled_point": pooled_point,
                "pooled_event": {
                    "attack_events": total_events,
                    "detected_attack_events": detected_events,
                    "event_recall": detected_events / max(1, total_events),
                    "mean_detection_delay_seconds_weighted": delay_weighted / max(1, detected_events) if detected_events else None,
                    "false_alarm_events": false_events,
                    "false_alarm_events_per_normal_hour": false_events / max(1e-12, normal_hours),
                },
                "per_file": file_rows,
            }
        result["models"][model_name] = by_quantile
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Leakage-free repeated-seed HAI threshold sensitivity audit")
    parser.add_argument("run_dirs", nargs="+", type=Path, help="Existing HAI runs whose seed-specific raw test scores will be reused")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "hai_21_03.json")
    parser.add_argument("--quantiles", nargs="+", type=float, default=[0.95, 0.975, 0.99, 0.995, 0.999])
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "summaries" / "hai21_threshold_sensitivity_seed42_46.json")
    args = parser.parse_args()
    runs = [evaluate_run(run_dir, args.config, args.quantiles) for run_dir in args.run_dirs]
    seeds = [run["seed"] for run in runs]
    if len(seeds) != len(set(seeds)):
        raise ValueError("each run must have a unique seed")
    report = {
        "study": "HAI 21.03 validation-quantile threshold sensitivity",
        "source_run_ids": [run["run_id"] for run in runs],
        "seeds": seeds,
        "quantiles": args.quantiles,
        "threshold_calibration": "each run's normal-only train sample and train3 validation scores; test scores and labels are never used to select thresholds",
        "test_score_source": "saved raw scores from each specified frozen seed run; detector models are refit only to reconstruct validation quantiles",
        "explanations_recomputed": False,
        "aggregation_note": "means and ranges summarize repeated model fits on the same five held-out files and attack events; they are not independent test replications or confidence intervals",
        "runs": runs,
        "mean_and_seed_range": aggregate_runs(runs),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    compact = {}
    for model, quantiles in report["mean_and_seed_range"].items():
        compact[model] = {}
        for q, values in quantiles.items():
            compact[model][q] = {
                key: values[key] for key in ["validation_fpr_observed", "event_recall", "false_alarm_events_per_normal_hour", "point_f1"]
            }
    print(json.dumps({"output": str(args.output), "seeds": seeds, "models": compact}, indent=2))


if __name__ == "__main__":
    main()
