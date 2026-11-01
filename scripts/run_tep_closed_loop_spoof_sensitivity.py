"""Compare unbounded and statistically bounded closed-loop TEP spoof profiles.

The bounded profiles use training-normal RobustZ scales as explicit stress
settings. They are not claims about instrument limits or field attacker ability.
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
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

import run_tep_closed_loop_spoof_pilot as shared  # noqa: E402
from xai_cps.detectors import RobustZDetector  # noqa: E402
from xai_cps.metrics import binary_metrics  # noqa: E402


def run(config: dict) -> dict:
    tep_simulator = shared._load_simulator()
    duration = float(config["duration_hours"])
    onset = float(config["event_onset_hours"])
    channel = int(config["spoof_channel_index_zero_based"])
    disturbance = {int(config["disturbance_id"]): (onset, 1)}

    # Clean-wrapper equivalence check before any result is recorded.
    check_duration = min(0.1, duration)
    native_check = shared._run_native(
        tep_simulator, int(config["train_seed"]), check_duration, None
    )
    wrapped_check, _, _, shared_view_equal = shared._run_shared_view(
        tep_simulator,
        int(config["train_seed"]),
        check_duration,
        None,
        onset,
    )
    np.testing.assert_array_equal(native_check.states, wrapped_check.states)
    np.testing.assert_array_equal(
        native_check.measurements, wrapped_check.measurements
    )
    np.testing.assert_array_equal(
        native_check.manipulated_vars, wrapped_check.manipulated_vars
    )

    train = shared._run_native(tep_simulator, int(config["train_seed"]), duration, None)
    validation = shared._run_native(
        tep_simulator, int(config["validation_seed"]), duration, None
    )
    train_x = np.asarray(train.measurements[:-1], dtype=float)
    validation_x = np.asarray(validation.measurements[:-1], dtype=float)
    detector = RobustZDetector().fit(
        train_x, shared._first_difference(train_x)
    )
    validation_scores = detector.score(
        validation_x, shared._first_difference(validation_x)
    )
    quantile = float(config["threshold_quantile"])
    threshold = float(np.quantile(validation_scores, quantile, method="higher"))
    reference = float(detector.x_ref[channel])
    scale = float(detector.x_scale[channel])
    event_onset_sample = int(round(onset * 3600.0))
    test_records = []

    for seed_value in config["test_seeds"]:
        seed = int(seed_value)
        clean_result = shared._run_native(
            tep_simulator, seed, duration, disturbance
        )
        clean_x = np.asarray(clean_result.measurements[:-1], dtype=float)
        clean_scores = detector.score(clean_x, shared._first_difference(clean_x))
        time_seconds = np.arange(len(clean_x), dtype=float)
        event_mask = time_seconds >= onset * 3600.0
        clean_metrics = binary_metrics(event_mask, clean_scores, threshold)
        clean_alarms = np.flatnonzero(event_mask & (clean_scores > threshold))
        clean_first = int(clean_alarms[0]) if clean_alarms.size else None
        clean_explanation = (
            shared._explain_at(detector, clean_x, clean_first, channel)
            if clean_first is not None
            else None
        )
        profile_records = []

        for configured_profile in config["attack_profiles"]:
            profile = dict(configured_profile)
            if profile["mode"] == "hold_at_training_reference":
                replacement = reference
                callback_profile = None
            elif profile["mode"] == "rate_limited_additive_bias":
                replacement = None
                profile["training_robust_scale"] = scale
                callback_profile = profile
            else:
                raise ValueError(f"Unsupported profile: {profile['mode']}")

            attacked_result, raw, observed, same_view = shared._run_shared_view(
                tep_simulator,
                seed,
                duration,
                disturbance,
                onset,
                spoof_channel=channel,
                spoof_value=replacement,
                spoof_profile=callback_profile,
            )
            attacked_scores = detector.score(
                observed, shared._first_difference(observed)
            )
            attacked_metrics = binary_metrics(event_mask, attacked_scores, threshold)
            attacked_alarms = np.flatnonzero(
                event_mask & (attacked_scores > threshold)
            )
            attacked_first = int(attacked_alarms[0]) if attacked_alarms.size else None
            attacked_explanation = (
                shared._explain_at(detector, observed, attacked_first, channel)
                if attacked_first is not None
                else None
            )
            at_clean_first = (
                {
                    "alarm_preserved_at_same_sample": bool(
                        attacked_scores[clean_first] > threshold
                    ),
                    "score": float(attacked_scores[clean_first]),
                    "explanation": shared._explain_at(
                        detector, observed, clean_first, channel
                    ),
                }
                if clean_first is not None
                else None
            )
            edits = (observed[:, channel] - raw[:, channel]) / scale
            event_edits = edits[event_mask]
            state_delta = np.asarray(attacked_result.states) - np.asarray(
                clean_result.states
            )
            mv_delta = np.asarray(attacked_result.manipulated_vars) - np.asarray(
                clean_result.manipulated_vars
            )
            profile_records.append(
                {
                    "name": profile["name"],
                    "mode": profile["mode"],
                    "parameters": {
                        key: value
                        for key, value in profile.items()
                        if key not in {"training_robust_scale"}
                    },
                    "clean_detection": clean_metrics,
                    "spoofed_detection": attacked_metrics,
                    "clean_first_alarm_sample": clean_first,
                    "spoofed_first_alarm_sample": attacked_first,
                    "clean_first_alarm_explanation": clean_explanation,
                    "spoofed_first_alarm_explanation": attacked_explanation,
                    "spoofed_input_at_clean_first_alarm": at_clean_first,
                    "measurement_edit": {
                        "channel": config["spoof_channel_name"],
                        "training_reference": reference,
                        "edit_at_event_onset_in_robust_scales": float(
                            edits[event_onset_sample]
                        ),
                        "edit_at_clean_first_alarm_in_robust_scales": (
                            float(edits[clean_first])
                            if clean_first is not None
                            else None
                        ),
                        "max_abs_edit_in_robust_scales": float(
                            np.max(np.abs(event_edits))
                        ),
                        "p95_abs_edit_in_robust_scales": float(
                            np.quantile(np.abs(event_edits), 0.95)
                        ),
                        "controller_and_ids_receive_same_observation": same_view,
                    },
                    "closed_loop_divergence_from_clean_same_seed": {
                        "max_abs_state_difference": float(
                            np.max(np.abs(state_delta))
                        ),
                        "max_abs_manipulated_variable_difference": float(
                            np.max(np.abs(mv_delta))
                        ),
                    },
                }
            )

        test_records.append(
            {
                "seed": seed,
                "clean_shutdown": bool(clean_result.shutdown),
                "clean_detection": clean_metrics,
                "clean_first_alarm_sample": clean_first,
                "clean_first_alarm_explanation": clean_explanation,
                "profiles": profile_records,
            }
        )

    return {
        "study": "TEP closed-loop sensor-spoof profile sensitivity pilot",
        "status": "exploratory; statistical bounds are stress settings, not field constraints",
        "simulator": {
            **config["simulator"],
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "protocol": {
            "duration_hours": duration,
            "event_onset_hours": onset,
            "disturbance": "IDV(6) A-feed loss process disturbance",
            "training_seed": int(config["train_seed"]),
            "validation_seed": int(config["validation_seed"]),
            "test_seeds": [int(seed) for seed in config["test_seeds"]],
            "screening_seeds_excluded": config[
                "screening_seeds_excluded_from_test"
            ],
            "detector": "normal-only RobustZ on 41 channels and first differences",
            "threshold_quantile": quantile,
            "threshold": threshold,
            "validation_false_positive_rate": float(
                np.mean(validation_scores > threshold)
            ),
            "channel": config["spoof_channel_name"],
            "reference": reference,
            "same_spoofed_observation_used_for_control_and_ids": True,
            "identity_wrapper_matches_native": True,
            "attack_profiles_fixed_before_test": config["attack_profiles"],
        },
        "test_runs": test_records,
        "limitations": [
            "IDV(6) is a process disturbance; the study overlays a sensor manipulation and does not reproduce a field exploit.",
            "Robust training scales and chosen slew rates are statistical stress settings, not instrument specifications or validated attacker capabilities.",
            "The process state is simulator ground truth, not a deployable integrity sensor.",
            "One detector, one channel, one disturbance, and five simulator seeds are evaluated; inference is descriptive only.",
            "No causal explanation, safety, or novelty claim is established.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs/tep_closed_loop_spoof_sensitivity.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT
        / "results/summaries/tep_closed_loop_spoof_sensitivity.json",
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
