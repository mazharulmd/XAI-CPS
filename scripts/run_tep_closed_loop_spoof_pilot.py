"""Evaluate a shared-view, closed-loop TEP sensor-spoof mechanism pilot.

Unlike run_tep_interposition_pilot.py, this script changes the measurement
before the TEP controller acts and scores that exact same observation stream
with the IDS after the run. The setup is an exploratory simulator experiment,
not a calibrated field threat or a novelty claim.
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


SIMULATOR_COMMIT = "9a6c8e5fcef4a2850778704e7793c87b0a187005"


def _first_difference(values: np.ndarray) -> np.ndarray:
    delta = np.zeros_like(values, dtype=float)
    delta[1:] = values[1:] - values[:-1]
    return delta


def _load_simulator():
    try:
        from tep import TEPSimulator
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise SystemExit(
            "Install the pinned optional TEP simulator as documented in "
            "docs/research/TEP_CLOSED_LOOP_SPOOF_PILOT.md."
        ) from exc
    return TEPSimulator


def _run_native(tep_simulator, seed: int, duration: float, disturbance: dict | None):
    sim = tep_simulator(random_seed=seed, backend="python")
    sim.initialize()
    return sim.simulate(
        duration_hours=duration,
        disturbances=disturbance,
        record_interval=1,
    )


def _run_shared_view(
    tep_simulator,
    seed: int,
    duration: float,
    disturbance: dict | None,
    onset_hours: float,
    spoof_channel: int | None = None,
    spoof_value: float | None = None,
):
    sim = tep_simulator(random_seed=seed, backend="python")
    sim.initialize()
    native_controller = sim.controller
    raw_measurements: list[np.ndarray] = []
    controller_observations: list[np.ndarray] = []
    ids_observations: list[np.ndarray] = []
    onset_step = int(round(onset_hours * 3600.0))

    def shared_observation_controller(xmeas, xmv, step):
        raw = np.array(xmeas, dtype=float, copy=True)
        observed = raw.copy()
        if spoof_channel is not None and step >= onset_step:
            observed[spoof_channel] = float(spoof_value)
        raw_measurements.append(raw)
        # This stored array is exactly the input used by the controller below
        # and the passive IDS observation stream evaluated after simulation.
        controller_input = observed.copy()
        ids_input = observed.copy()
        controller_observations.append(controller_input)
        ids_observations.append(ids_input)
        return native_controller.calculate(controller_input, xmv, step)

    result = sim.simulate_with_controller(
        duration_hours=duration,
        controller=shared_observation_controller,
        disturbances=disturbance,
        record_interval=1,
    )
    raw = np.asarray(raw_measurements, dtype=float)
    controller_trace = np.asarray(controller_observations, dtype=float)
    ids_trace = np.asarray(ids_observations, dtype=float)
    if raw.ndim != 2 or controller_trace.shape != raw.shape or ids_trace.shape != raw.shape:
        raise ValueError("The simulator did not return aligned callback observations")
    if raw.shape[1] != 41 or len(result.time) != len(ids_trace) + 1:
        raise ValueError(
            "Unexpected TEP callback/result alignment: expected 41 channels "
            "and one final post-control result sample"
        )
    # Simulator result row t is the raw process measurement at controller
    # sample t, with one final post-integration row after the final callback.
    np.testing.assert_allclose(raw, result.measurements[:-1], rtol=0.0, atol=0.0)
    same_view = bool(np.array_equal(controller_trace, ids_trace))
    if not same_view:
        raise AssertionError("Controller and IDS observation paths diverged")
    return result, raw, ids_trace, same_view


def _rank(contributions: np.ndarray, feature_index: int) -> int:
    order = np.argsort(-contributions, kind="mergesort")
    return int(np.flatnonzero(order == feature_index)[0]) + 1


def _explain_at(detector, x: np.ndarray, point: int, channel: int) -> dict:
    dx = _first_difference(x)
    names = [f"XMEAS({i + 1:02d})" for i in range(x.shape[1])]
    values, _ = grouped_ablation(
        detector,
        x[point],
        dx[point],
        names,
        previous_x=x[point - 1] if point else None,
    )
    values = np.asarray(values, dtype=float)
    return {
        "point_index": int(point),
        "target_rank": _rank(values, channel),
        "top5_channels": [names[i] for i in np.argsort(-values, kind="mergesort")[:5]],
        "target_ablation_value": float(values[channel]),
        "contribution_sum": float(values.sum()),
    }


def run(config: dict) -> dict:
    tep_simulator = _load_simulator()
    duration = float(config["duration_hours"])
    onset = float(config["event_onset_hours"])
    channel = int(config["spoof_channel_index_zero_based"])
    disturbance_id = int(config["disturbance_id"])
    if not 0 <= channel < 41:
        raise ValueError("spoof channel must be a zero-based TEP XMEAS index")
    if config["simulator"]["commit"] != SIMULATOR_COMMIT:
        raise ValueError("Configuration simulator commit does not match this runner")

    # Validate that instrumentation alone preserves the native controller path.
    equivalence_duration = min(0.1, duration)
    native_equivalence = _run_native(
        tep_simulator, int(config["train_seed"]), equivalence_duration, None
    )
    wrapped_equivalence, _, _, equivalence_shared_view = _run_shared_view(
        tep_simulator,
        int(config["train_seed"]),
        equivalence_duration,
        None,
        onset,
    )
    np.testing.assert_array_equal(native_equivalence.states, wrapped_equivalence.states)
    np.testing.assert_array_equal(
        native_equivalence.measurements, wrapped_equivalence.measurements
    )
    np.testing.assert_array_equal(
        native_equivalence.manipulated_vars, wrapped_equivalence.manipulated_vars
    )

    # Detector fitting and threshold selection use complete normal runs only.
    train = _run_native(tep_simulator, int(config["train_seed"]), duration, None)
    validation = _run_native(
        tep_simulator, int(config["validation_seed"]), duration, None
    )
    train_x = np.asarray(train.measurements[:-1], dtype=float)
    validation_x = np.asarray(validation.measurements[:-1], dtype=float)
    detector = RobustZDetector().fit(train_x, _first_difference(train_x))
    quantile = float(config["threshold_quantile"])
    validation_scores = detector.score(
        validation_x, _first_difference(validation_x)
    )
    threshold = float(np.quantile(validation_scores, quantile, method="higher"))
    reference = float(detector.x_ref[channel])
    disturbance = {disturbance_id: (onset, 1)}
    test_records = []

    for seed_value in config["test_seeds"]:
        seed = int(seed_value)
        clean_result = _run_native(tep_simulator, seed, duration, disturbance)
        clean_x = np.asarray(clean_result.measurements[:-1], dtype=float)
        attacked_result, raw, observed, shared_view_equal = _run_shared_view(
            tep_simulator,
            seed,
            duration,
            disturbance,
            onset,
            spoof_channel=channel,
            spoof_value=reference,
        )
        if clean_x.shape != observed.shape:
            raise ValueError("Clean and spoofed process runs have inconsistent lengths")

        clean_scores = detector.score(clean_x, _first_difference(clean_x))
        attacked_scores = detector.score(observed, _first_difference(observed))
        controller_ids_equal = shared_view_equal
        seconds = np.arange(len(observed), dtype=float)
        event_mask = seconds >= onset * 3600.0
        labels = event_mask
        clean_metrics = binary_metrics(labels, clean_scores, threshold)
        attacked_metrics = binary_metrics(labels, attacked_scores, threshold)
        clean_alarms = np.flatnonzero(event_mask & (clean_scores > threshold))
        attacked_alarms = np.flatnonzero(event_mask & (attacked_scores > threshold))
        clean_first = int(clean_alarms[0]) if clean_alarms.size else None
        attacked_first = int(attacked_alarms[0]) if attacked_alarms.size else None
        clean_explanation = (
            _explain_at(detector, clean_x, clean_first, channel)
            if clean_first is not None
            else None
        )
        attacked_explanation = (
            _explain_at(detector, observed, attacked_first, channel)
            if attacked_first is not None
            else None
        )
        attacked_at_clean_first = (
            {
                "alarm_preserved_at_same_sample": bool(
                    attacked_scores[clean_first] > threshold
                ),
                "score": float(attacked_scores[clean_first]),
                "explanation": _explain_at(detector, observed, clean_first, channel),
            }
            if clean_first is not None
            else None
        )
        # The two runs share the same seed and disturbance. Divergence is a
        # simulator diagnostic for the effect of controller-observation spoofing.
        state_delta = np.asarray(attacked_result.states) - np.asarray(clean_result.states)
        mv_delta = np.asarray(attacked_result.manipulated_vars) - np.asarray(
            clean_result.manipulated_vars
        )
        edit_scales = np.abs(observed[:, channel] - raw[:, channel]) / max(
            float(detector.x_scale[channel]), 1e-12
        )
        edit_at_clean_first = (
            float(abs(observed[clean_first, channel] - raw[clean_first, channel]))
            if clean_first is not None
            else None
        )
        edit_scales_at_clean_first = (
            float(edit_at_clean_first / max(float(detector.x_scale[channel]), 1e-12))
            if edit_at_clean_first is not None
            else None
        )
        test_records.append(
            {
                "seed": seed,
                "sample_count": int(len(observed)),
                "shutdown_clean": bool(clean_result.shutdown),
                "shutdown_spoofed": bool(attacked_result.shutdown),
                "process_disturbance": "IDV(6) A-feed loss",
                "sensor_spoof": {
                    "channel": config["spoof_channel_name"],
                    "mode": "hold_at_normal_training_reference",
                    "reference_value": reference,
                    "first_difference_from_raw_at_onset": float(
                        observed[int(onset * 3600), channel]
                        - raw[int(onset * 3600), channel]
                    ),
                    "max_abs_edit_in_robust_scales": float(np.max(edit_scales)),
                    "p95_abs_edit_in_robust_scales": float(
                        np.quantile(edit_scales[event_mask], 0.95)
                    ),
                    "abs_edit_at_clean_first_alarm": edit_at_clean_first,
                    "edit_at_clean_first_alarm_in_robust_scales": edit_scales_at_clean_first,
                    "controller_and_ids_receive_same_observation": controller_ids_equal,
                },
                "clean_detection": clean_metrics,
                "spoofed_detection": attacked_metrics,
                "clean_first_alarm_sample": clean_first,
                "spoofed_first_alarm_sample": attacked_first,
                "clean_first_alarm_delay_seconds": (
                    float(clean_first - onset * 3600.0)
                    if clean_first is not None
                    else None
                ),
                "spoofed_first_alarm_delay_seconds": (
                    float(attacked_first - onset * 3600.0)
                    if attacked_first is not None
                    else None
                ),
                "clean_first_alarm_explanation": clean_explanation,
                "spoofed_first_alarm_explanation": attacked_explanation,
                "spoofed_input_at_clean_first_alarm": attacked_at_clean_first,
                "closed_loop_divergence_from_clean_same_seed": {
                    "max_abs_state_difference": float(np.max(np.abs(state_delta))),
                    "terminal_state_l2_difference": float(
                        np.linalg.norm(state_delta[-1])
                    ),
                    "max_abs_manipulated_variable_difference": float(
                        np.max(np.abs(mv_delta))
                    ),
                },
            }
        )

    return {
        "study": "TEP closed-loop sensor-spoofing mechanism pilot",
        "status": "exploratory; not a novelty or field-feasibility claim",
        "simulator": {
            **config["simulator"],
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "protocol": {
            "duration_hours": duration,
            "event_onset_hours": onset,
            "disturbance": "IDV(6) A-feed loss process disturbance",
            "spoof_channel": config["spoof_channel_name"],
            "spoof_mode": config["spoof_mode"],
            "same_spoofed_observation_used_for_control_and_ids": True,
            "ids_scoring": "offline replay of exact controller observation trace; passive, no feedback to process",
            "training_seed": int(config["train_seed"]),
            "validation_seed": int(config["validation_seed"]),
            "test_seeds": [int(s) for s in config["test_seeds"]],
            "threshold_quantile": quantile,
            "threshold": threshold,
            "validation_false_positive_rate": float(
                np.mean(validation_scores > threshold)
            ),
            "normal_training_reference_value": reference,
            "screening_seeds_excluded": config.get(
                "screening_seeds_excluded_from_test", []
            ),
            "controller_path_compromised": True,
        },
        "instrumentation_check": {
            "duration_hours": equivalence_duration,
            "identity_controller_matches_native_state_measurement_and_mv_arrays": True,
            "controller_and_ids_observation_traces_equal": equivalence_shared_view,
        },
        "test_runs": test_records,
        "limitations": [
            "IDV(6) is a simulator process disturbance; this does not model a field adversary or a complete exploit chain.",
            "The sensor replacement is a constant training reference, without calibrated sensor bounds, noise, or slew constraints.",
            "The process state is available only as simulator ground truth and is not a deployable integrity signal by itself.",
            "Only one detector, one channel, one disturbance, and five held-out simulator seeds are evaluated.",
            "The scenario is selected to check shared measurement interposition and does not establish novelty or causal explanation validity.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs/tep_closed_loop_spoof_pilot.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT
        / "results/summaries/tep_closed_loop_spoof_pilot.json",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = run(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
