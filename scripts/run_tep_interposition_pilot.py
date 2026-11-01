"""Run a small, leakage-separated TEP IDS/explanation-steering pilot.

This is an exploratory white-box stress test, not a confirmatory experiment.
The sensor edit is applied to one IDS observation only; the TEP controller and
plant trajectory remain unmodified during the test run. See the generated
summary and docs/research/TEP_INTERPOSITION_PILOT.md for scope and caveats.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import sys

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from xai_cps.detectors import RobustZDetector  # noqa: E402
from xai_cps.explainers import grouped_ablation  # noqa: E402
from xai_cps.metrics import binary_metrics  # noqa: E402


def _load_simulator():
    try:
        from tep import TEPSimulator
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "Install the pinned optional TEP package using the command in "
            "docs/research/TEP_INTERPOSITION_PILOT.md."
        ) from exc
    return TEPSimulator


def _first_difference(values: np.ndarray) -> np.ndarray:
    delta = np.zeros_like(values, dtype=float)
    delta[1:] = values[1:] - values[:-1]
    return delta


def _simulate(tep_simulator, seed: int, duration_hours: float,
              disturbance: dict[int, tuple[float, int]] | None = None):
    sim = tep_simulator(random_seed=seed, backend="python")
    sim.initialize()
    return sim.simulate(
        duration_hours=duration_hours,
        disturbances=disturbance,
        record_interval=1,
    )


def _rank(values: np.ndarray, index: int) -> int:
    order = np.argsort(-np.asarray(values), kind="mergesort")
    return int(np.flatnonzero(order == index)[0]) + 1


def _explain(detector, x: np.ndarray, previous: np.ndarray,
             feature_names: list[str]) -> np.ndarray:
    dx = x - previous
    values, _ = grouped_ablation(
        detector,
        x,
        dx,
        feature_names,
        previous_x=previous,
    )
    return np.asarray(values, dtype=float)


def _steering_search(detector, x: np.ndarray, previous: np.ndarray,
                     threshold: float, feature_names: list[str],
                     target_index: int, multipliers: list[float]) -> dict:
    clean_dx = x - previous
    clean_score = float(detector.score(x[None, :], clean_dx[None, :])[0])
    clean_values = _explain(detector, x, previous, feature_names)
    clean_rank = _rank(clean_values, target_index)
    eligible = []
    total_candidates = 0

    # White-box, target-aware candidate sweep: one altered measurement at one
    # IDS observation. The prior observation is fixed, so dx remains x_t-x_{t-1}.
    for channel in range(len(x)):
        for multiplier in multipliers:
            total_candidates += 1
            altered = x.copy()
            altered[channel] += multiplier * detector.x_scale[channel]
            altered_dx = altered - previous
            score = float(detector.score(altered[None, :], altered_dx[None, :])[0])
            if score <= threshold:
                continue
            values = _explain(detector, altered, previous, feature_names)
            rank = _rank(values, target_index)
            eligible.append({
                "channel_index_zero_based": int(channel),
                "channel_name": feature_names[channel].split(".", 1)[0],
                "robust_scale_multiplier": float(multiplier),
                "normalized_absolute_edit": float(abs(multiplier)),
                "alarm_score": score,
                "target_rank": rank,
                "target_ablation_value": float(values[target_index]),
                "top5_channels": [
                    feature_names[i].split(".", 1)[0]
                    for i in np.argsort(-values, kind="mergesort")[:5]
                ],
            })

    if not eligible:
        best = None
        best_rank = clean_rank
    else:
        # Maximize demotion of the predeclared target; break ties by smaller
        # edit and then by lower score increase. No test-label metric is optimized.
        best = min(
            eligible,
            key=lambda item: (
                -item["target_rank"],
                item["normalized_absolute_edit"],
                abs(item["alarm_score"] - clean_score),
                item["channel_index_zero_based"],
            ),
        )
        best_rank = int(best["target_rank"])

    return {
        "clean_alarm_score": clean_score,
        "threshold": float(threshold),
        "alarm_preserved_for_clean_input": bool(clean_score > threshold),
        "clean_target_rank": clean_rank,
        "clean_target_ablation_value": float(clean_values[target_index]),
        "clean_top5_channels": [
            feature_names[i].split(".", 1)[0]
            for i in np.argsort(-clean_values, kind="mergesort")[:5]
        ],
        "candidate_count": total_candidates,
        "alarm_preserving_candidate_count": len(eligible),
        "best_alarm_preserving_candidate": best,
        "target_rank_demoted": bool(best is not None and best_rank > clean_rank),
        "target_rank_change": int(best_rank - clean_rank),
    }


def _persistent_steering_search(
    detector,
    measurements: np.ndarray,
    event_mask: np.ndarray,
    first_baseline_alarm: int,
    threshold: float,
    feature_names: list[str],
    target_index: int,
    multipliers: list[float],
) -> dict:
    """Search one-channel IDS-only hold/bias attacks over the event suffix."""
    base_previous = measurements[first_baseline_alarm - 1]
    base_values = _explain(
        detector,
        measurements[first_baseline_alarm],
        base_previous,
        feature_names,
    )
    base_rank = _rank(base_values, target_index)
    candidates = []
    onset_index = int(np.flatnonzero(event_mask)[0])
    suffix = np.flatnonzero(event_mask)

    def evaluate(mode: str, channel: int, multiplier: float | None) -> None:
        altered = measurements.copy()
        if mode == "hold_at_training_reference":
            altered[suffix, channel] = detector.x_ref[channel]
        elif mode == "persistent_additive_bias":
            assert multiplier is not None
            altered[suffix, channel] += multiplier * detector.x_scale[channel]
        else:  # pragma: no cover - internal invariant
            raise ValueError(mode)

        altered_dx = _first_difference(altered)
        altered_scores = detector.score(altered, altered_dx)
        score_at_baseline_alarm = float(altered_scores[first_baseline_alarm])
        if score_at_baseline_alarm <= threshold:
            return
        altered_previous = altered[first_baseline_alarm - 1]
        values = _explain(
            detector,
            altered[first_baseline_alarm],
            altered_previous,
            feature_names,
        )
        if mode == "hold_at_training_reference":
            normalized_edit = float(
                abs(measurements[first_baseline_alarm, channel] - detector.x_ref[channel])
                / detector.x_scale[channel]
            )
        else:
            normalized_edit = float(abs(multiplier))
        candidates.append({
            "mode": mode,
            "channel_index_zero_based": int(channel),
            "channel_name": feature_names[channel].split(".", 1)[0],
            "robust_scale_multiplier": multiplier,
            "normalized_edit_at_baseline_alarm": normalized_edit,
            "alarm_score_at_baseline_alarm": score_at_baseline_alarm,
            "event_alarm_rate_after_attack": float(
                np.mean(altered_scores[event_mask] > threshold)
            ),
            "target_rank": _rank(values, target_index),
            "target_ablation_value": float(values[target_index]),
            "top5_channels": [
                feature_names[i].split(".", 1)[0]
                for i in np.argsort(-values, kind="mergesort")[:5]
            ],
            "alarm_preserved_at_same_sample": True,
        })

    for channel in range(measurements.shape[1]):
        evaluate("hold_at_training_reference", channel, None)
        for multiplier in multipliers:
            evaluate("persistent_additive_bias", channel, multiplier)

    best = None
    if candidates:
        best = min(
            candidates,
            key=lambda item: (
                -item["target_rank"],
                item["normalized_edit_at_baseline_alarm"],
                item["channel_index_zero_based"],
                item["mode"],
            ),
        )
    best_rank = base_rank if best is None else int(best["target_rank"])
    rank_histogram: dict[str, int] = {}
    for item in candidates:
        key = str(item["target_rank"])
        rank_histogram[key] = rank_histogram.get(key, 0) + 1
    event_alarm_rates = [item["event_alarm_rate_after_attack"] for item in candidates]
    return {
        "timing": "persistent IDS-input transformation from event onset through the end of the run",
        "onset_sample_index": onset_index,
        "baseline_first_alarm_sample_index": int(first_baseline_alarm),
        "clean_target_rank": base_rank,
        "clean_target_ablation_value": float(base_values[target_index]),
        "clean_top5_channels": [
            feature_names[i].split(".", 1)[0]
            for i in np.argsort(-base_values, kind="mergesort")[:5]
        ],
        "candidate_count": measurements.shape[1] * (1 + len(multipliers)),
        "alarm_preserving_candidate_count": len(candidates),
        "target_rank_histogram_over_alarm_preserving_candidates": rank_histogram,
        "event_alarm_rate_quantiles_over_alarm_preserving_candidates": (
            None if not event_alarm_rates else {
                "min": float(np.min(event_alarm_rates)),
                "median": float(np.median(event_alarm_rates)),
                "max": float(np.max(event_alarm_rates)),
            }
        ),
        "best_alarm_preserving_candidate": best,
        "target_rank_demoted": bool(best is not None and best_rank > base_rank),
        "target_rank_change": int(best_rank - base_rank),
    }


def run(config: dict) -> dict:
    tep_simulator = _load_simulator()
    duration = float(config["duration_hours"])
    onset = float(config["event_onset_hours"])
    train = _simulate(tep_simulator, int(config["train_seed"]), duration)
    validation = _simulate(tep_simulator, int(config["validation_seed"]), duration)
    train_x = np.asarray(train.measurements, dtype=float)
    validation_x = np.asarray(validation.measurements, dtype=float)
    if train_x.shape[1] != 41 or validation_x.shape[1] != 41:
        raise ValueError("Expected the pinned TEP API's 41 measurement channels")

    detector = RobustZDetector().fit(train_x, _first_difference(train_x))
    validation_scores = detector.score(validation_x, _first_difference(validation_x))
    quantile = float(config["threshold_quantile"])
    threshold = float(np.quantile(validation_scores, quantile, method="higher"))
    feature_names = [f"XMEAS{i + 1:02d}.channel" for i in range(train_x.shape[1])]
    target_index = int(config["target_channel_index_zero_based"])
    if not 0 <= target_index < train_x.shape[1]:
        raise ValueError("Configured target index is outside the measurement vector")

    test_runs = []
    disturbance_id = int(config["disturbance_id"])
    multipliers = [
        float(v)
        for v in config["attack_search"]["robust_scale_multipliers"]
    ]
    persistent_multipliers = [
        float(v)
        for v in config["persistent_attack_search"]["robust_scale_multipliers"]
    ]
    for seed in config["test_seeds"]:
        result = _simulate(
            tep_simulator,
            int(seed),
            duration,
            disturbance={disturbance_id: (onset, 1)},
        )
        measurements = np.asarray(result.measurements, dtype=float)
        scores = detector.score(measurements, _first_difference(measurements))
        event_mask = np.asarray(result.time, dtype=float) >= onset
        metrics = binary_metrics(event_mask, scores, threshold)
        alarms = np.flatnonzero(event_mask & (scores > threshold))
        if alarms.size:
            first = int(alarms[0])
            prior = measurements[max(0, first - 1)]
            explanation = _steering_search(
                detector,
                measurements[first],
                prior,
                threshold,
                feature_names,
                target_index,
                multipliers,
            )
            persistent_explanation = _persistent_steering_search(
                detector,
                measurements,
                event_mask,
                first,
                threshold,
                feature_names,
                target_index,
                persistent_multipliers,
            )
            detection_time = float(result.time[first])
            delay_seconds = max(0.0, (detection_time - onset) * 3600.0)
        else:
            first = None
            detection_time = None
            delay_seconds = None
            explanation = None
            persistent_explanation = None
        test_runs.append({
            "seed": int(seed),
            "sample_count": int(len(result.time)),
            "shutdown": bool(result.shutdown),
            "shutdown_time_hours": (
                None if result.shutdown_time is None else float(result.shutdown_time)
            ),
            "pre_event_alarm_rate": float(np.mean(scores[~event_mask] > threshold)),
            "event_alarm_rate": float(np.mean(scores[event_mask] > threshold)),
            "first_event_alarm_sample_index": first,
            "first_event_alarm_time_hours": detection_time,
            "event_detection_delay_seconds": delay_seconds,
            "point_metrics": metrics,
            "first_alarm_explanation_and_steering": explanation,
            "persistent_alarm_preserving_steering": persistent_explanation,
        })

    val_fpr = float(np.mean(validation_scores > threshold))
    return {
        "study": "TEP measurement-path explanation-steering pilot",
        "status": "exploratory; no novelty or physical-feasibility claim",
        "simulator": config["simulator"],
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "data_protocol": {
            "train_seed": int(config["train_seed"]),
            "validation_seed": int(config["validation_seed"]),
            "screening_seeds_excluded_from_test": [
                int(x) for x in config["screening_seeds_excluded_from_test"]
            ],
            "test_seeds": [int(x) for x in config["test_seeds"]],
            "duration_hours_per_run": duration,
            "samples_per_hour": 3600,
            "complete_simulation_runs_are_split_by_seed": True,
            "preprocessing_fit_on_train_only": True,
            "threshold_fit_on_validation_normal_only": True,
            "threshold_quantile": quantile,
            "validation_normal_alarm_rate": val_fpr,
        },
        "detector": {
            "name": "RobustZDetector",
            "features": "41 process measurements and representation-consistent first differences",
            "threshold": threshold,
            "explanation": "black-box single-channel ablation with training references; first differences recomputed from the fixed preceding observation",
        },
        "event": {
            "disturbance_id": disturbance_id,
            "disturbance_name": "IDV(6), A Feed Loss",
            "onset_hours": onset,
            "target_channel": config["target_channel_name"],
            "target_channel_index_zero_based": target_index,
        },
        "sensor_attack": {
            "path": "IDS observation only; controller and simulated plant are not re-run under the altered value",
            "attacker_knows_fitted_detector_and_threshold": bool(config["attack_search"]["attacker_knows_fitted_detector_and_threshold"]),
            "attacker_knows_event_target": bool(config["attack_search"]["attacker_knows_event_target"]),
            "physical_range_or_rate_constraint": False,
            "single_observation_policy": {
                "timing": "one IDS observation at the first post-onset alarm",
                "budget": "one channel, at most one robust training scale, tested at the configured discrete multipliers",
                "optimizer": "exhaustive post-hoc oracle sweep over channel and signed offset candidates; upper-bound stress test, not a deployable attack algorithm",
            },
            "persistent_policy": {
                "timing": "from event onset through the end of the run",
                "modes": config["persistent_attack_search"]["modes"],
                "attacker_knows_event_onset": bool(config["persistent_attack_search"]["attacker_knows_event_onset"]),
                "bias_budget": "one channel; constant offset no larger than one robust training scale",
                "reference_hold_constraint": "hold-at-training-reference candidates are not clipped to one robust scale; the realized normalized edit is reported",
            },
        },
        "test_runs": test_runs,
        "limitations": [
            "IDV(6) was selected after exploratory screening; seeds 44 and 45 were inspected during that selection and are excluded from the reported test seeds.",
            "One disturbance type, one detector family, two test seeds, and one target channel are insufficient for confirmatory inference.",
            "The one-observation IDS-path edit does not alter the controller input or plant trajectory.",
            "The robust-scale edit budget is not calibrated to sensor physics, instrument noise, reachability, or safety limits.",
            "The oracle search knows the event target and is a red-team upper bound.",
            "Single-channel ablation is an operational sensitivity measure, not a causal explanation or ground-truth proof.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "tep_interposition_pilot.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "results" / "summaries" / "tep_interposition_pilot.json",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = run(config)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
