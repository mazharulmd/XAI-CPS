from __future__ import annotations

import numpy as np


def validation_thresholds(scores: np.ndarray, quantiles: list[float]) -> dict[str, float]:
    """Calibrate thresholds from validation scores only using NumPy's linear quantile."""
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("validation scores must be a nonempty finite vector")
    result = {}
    for q in quantiles:
        q = float(q)
        if not 0.0 < q < 1.0:
            raise ValueError("quantiles must lie strictly between zero and one")
        result[f"{q:.6f}"] = float(np.quantile(values, q, method="linear"))
    return result


def aggregate_runs(runs: list[dict]) -> dict:
    """Summarize repeated fits over shared test events using means and ranges only."""
    metric_paths = {
        "validation_fpr_observed": ("validation_fpr_observed",),
        "point_precision": ("pooled_point", "precision"),
        "point_recall": ("pooled_point", "recall"),
        "point_f1": ("pooled_point", "f1"),
        "point_average_precision": ("pooled_point", "average_precision"),
        "test_point_fpr": ("pooled_point", "false_positive_rate"),
        "event_recall": ("pooled_event", "event_recall"),
        "mean_detection_delay_seconds": ("pooled_event", "mean_detection_delay_seconds_weighted"),
        "false_alarm_events_per_normal_hour": ("pooled_event", "false_alarm_events_per_normal_hour"),
    }
    aggregate = {}
    for model in sorted(runs[0]["models"]):
        aggregate[model] = {}
        for q_key in runs[0]["models"][model]:
            row = {"n_repeated_fits": len(runs)}
            thresholds = [run["models"][model][q_key]["threshold"] for run in runs]
            row["threshold_mean"] = float(np.mean(thresholds))
            row["threshold_seed_range"] = [float(min(thresholds)), float(max(thresholds))]
            for name, path in metric_paths.items():
                values = []
                for run in runs:
                    value = run["models"][model][q_key]
                    for key in path:
                        value = value[key]
                    if value is not None and np.isfinite(value):
                        values.append(float(value))
                row[name] = {
                    "mean": float(np.mean(values)) if values else None,
                    "seed_range": [float(min(values)), float(max(values))] if values else None,
                }
            row["attack_events_per_run"] = int(runs[0]["models"][model][q_key]["pooled_event"]["attack_events"])
            aggregate[model][q_key] = row
    return aggregate
