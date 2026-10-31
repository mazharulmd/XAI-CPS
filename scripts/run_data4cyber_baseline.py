from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import hashlib
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd

from xai_cps.data4cyber import (Data4CyberSplit, fit_normal_median_imputer,
                                load_data4cyber_benign_file, load_data4cyber_scenario)
from xai_cps.detectors import PCAResidualDetector, RobustZDetector
from xai_cps.explainers import channel_groups, component_group_ablation
from xai_cps.data import contiguous_events
from xai_cps.metrics import binary_metrics, event_metrics


def sequence_delta(values: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    delta = np.zeros_like(values, dtype=float)
    if len(values) < 2:
        return delta
    times_ns = pd.DatetimeIndex(timestamps).asi8
    seconds = np.diff(times_ns) / 1_000_000_000
    adjacent = (seconds > 0) & (seconds <= 1.5)
    delta[1:][adjacent] = values[1:][adjacent] - values[:-1][adjacent]
    return delta


def threshold_at_fpr(scores: np.ndarray, target: float) -> float:
    if not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Calibration scores are empty or non-finite")
    return float(np.quantile(scores, 1.0 - target, method="higher"))


def read_benign(path: Path) -> Data4CyberSplit:
    return load_data4cyber_benign_file(path)


def combine_test(attack: Data4CyberSplit, benign: Data4CyberSplit):
    frames = []
    for split in (attack, benign):
        frame = split.features.copy()
        frame["__timestamp"] = split.timestamps
        frame["__attack"] = split.attack
        frame["__phase"] = split.phase
        frame["__phase_all"] = split.phase_all
        frames.append(frame)
    joined = pd.concat(frames, ignore_index=True).sort_values("__timestamp", kind="mergesort")
    if joined["__timestamp"].duplicated().any():
        raise ValueError(f"Duplicate timestamps in test scenario {attack.scenario}")
    timestamps = pd.to_datetime(joined.pop("__timestamp"), utc=True).dt.tz_localize(None).to_numpy()
    labels = joined.pop("__attack").to_numpy(dtype=bool)
    phases = joined.pop("__phase").astype(str).to_numpy()
    phases_all = joined.pop("__phase_all").astype(str).to_numpy()
    features = joined
    return features, labels, timestamps, phases, phases_all


def random_hit_probability(n_groups: int, n_targets: int, k: int) -> float:
    """Probability that a uniformly random top-k set intersects the target set."""
    if n_groups <= 0 or n_targets <= 0:
        return 0.0
    k = min(k, n_groups)
    return 1.0 - math.comb(n_groups - n_targets, k) / math.comb(n_groups, k)


def explanation_for_attack_events(model, scores: np.ndarray, threshold: float, x: np.ndarray,
                                  dx: np.ndarray, y: np.ndarray, timestamps: np.ndarray,
                                  phases: np.ndarray, phases_all: np.ndarray,
                                  feature_names: list[str], target_groups: list[str],
                                  evidence_tier: str, target_status: str, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    results = []
    groups = channel_groups(feature_names)
    for event_index, (start, end) in enumerate(contiguous_events(y)):
        event_phase = str(phases_all[start]) or str(phases[start])
        group_universe = list(dict.fromkeys(groups))
        phase_lower = (event_phase + " " + str(phases[start])).lower()
        target_eligible = "recon" not in phase_lower
        alarm_indices = np.flatnonzero(scores[start:end] > threshold)
        alarm_index = int(start + alarm_indices[0]) if alarm_indices.size else None
        ranking: list[str] = []
        native_ranking: list[str] = []
        ranked_scores: dict[str, float] = {}
        perturb_jaccards: dict[float, list[float]] = {0.01: [], 0.05: []}
        native_perturb_jaccards: dict[float, list[float]] = {0.01: [], 0.05: []}
        explanation_ms = None
        fidelity_drop = None
        if alarm_index is not None:
            begin = time.perf_counter()
            ranked_scores = component_group_ablation(
                model, x[alarm_index], dx[alarm_index], feature_names,
                previous_x=x[alarm_index] - dx[alarm_index])
            explanation_ms = (time.perf_counter() - begin) * 1000.0
            ranking = sorted(ranked_scores, key=lambda key: (-ranked_scores[key], key))
            native_scores = np.asarray(model.contributions(
                x[alarm_index][None, :], dx[alarm_index][None, :])[0], dtype=float)
            native_group_scores = {
                group: float(np.mean(native_scores[np.asarray(groups) == group]))
                for group in dict.fromkeys(groups)
            }
            native_ranking = sorted(native_group_scores, key=lambda key: (-native_group_scores[key], key))
            if ranking:
                fidelity_drop = float(ranked_scores[ranking[0]])
            baseline_top = set(ranking[:5])
            native_baseline_top = set(native_ranking[:5])
            previous = x[alarm_index] - dx[alarm_index]
            for sigma in perturb_jaccards:
                for _ in range(5):
                    perturbed = x[alarm_index] + rng.normal(0.0, sigma * model.x_scale, size=x.shape[1])
                    perturbed_dx = perturbed - previous
                    perturb_scores = component_group_ablation(
                        model, perturbed, perturbed_dx, feature_names, previous_x=previous)
                    perturbed_top = set(sorted(perturb_scores, key=lambda key: (-perturb_scores[key], key))[:5])
                    union = baseline_top | perturbed_top
                    perturb_jaccards[sigma].append(len(baseline_top & perturbed_top) / max(1, len(union)))
                    perturbed_native = np.asarray(model.contributions(
                        perturbed[None, :], perturbed_dx[None, :])[0], dtype=float)
                    perturbed_native_groups = {
                        group: float(np.mean(perturbed_native[np.asarray(groups) == group]))
                        for group in dict.fromkeys(groups)
                    }
                    perturbed_native_top = set(sorted(perturbed_native_groups,
                                                      key=lambda key: (-perturbed_native_groups[key], key))[:5])
                    native_union = native_baseline_top | perturbed_native_top
                    native_perturb_jaccards[sigma].append(
                        len(native_baseline_top & perturbed_native_top) / max(1, len(native_union)))
        result = {
            "attack_event_index": event_index,
            "event_start_utc": str(timestamps[start]),
            "event_last_sample_utc": str(timestamps[end - 1]),
            "phase_at_event_start": str(phases[start]),
            "phase_all_at_event_start": event_phase,
            "target_evaluation_eligible": target_eligible,
            "target_groups": target_groups if target_eligible else [],
            "evidence_tier": evidence_tier if target_eligible else "phase_only_reconnaissance",
            "target_status": target_status if target_eligible else "not_applicable",
            "detected": alarm_index is not None,
            "first_alarm_utc": str(timestamps[alarm_index]) if alarm_index is not None else None,
            "phase_at_first_alarm": str(phases[alarm_index]) if alarm_index is not None else None,
            "seconds_from_event_start_to_first_alarm": (
                float((timestamps[alarm_index] - timestamps[start]) / np.timedelta64(1, "s"))
                if alarm_index is not None else None),
            "ranked_groups": ranking,
            "group_ablation_ranked_groups": ranking,
            "native_group_mean_ranked_groups": native_ranking,
            "group_ablation_scores": ranked_scores,
            "target_hit_at_1": bool(alarm_index is not None and target_eligible
                                     and any(group in target_groups for group in ranking[:1])),
            "target_hit_at_3": bool(alarm_index is not None and target_eligible
                                     and any(group in target_groups for group in ranking[:3])),
            "target_hit_at_5": bool(alarm_index is not None and target_eligible
                                     and any(group in target_groups for group in ranking[:5])),
            "native_target_hit_at_1": bool(alarm_index is not None and target_eligible
                                            and any(group in target_groups for group in native_ranking[:1])),
            "native_target_hit_at_3": bool(alarm_index is not None and target_eligible
                                            and any(group in target_groups for group in native_ranking[:3])),
            "native_target_hit_at_5": bool(alarm_index is not None and target_eligible
                                            and any(group in target_groups for group in native_ranking[:5])),
            "random_reference_unconditional_hit_probabilities": {
                str(k): (random_hit_probability(len(group_universe),
                        len(set(target_groups) & set(group_universe)), k) if alarm_index is not None and target_eligible else 0.0)
                for k in (1, 3, 5)
            },
            "top_group_score_reduction": fidelity_drop,
            "top5_jaccard_under_measurement_noise": {
                str(sigma): float(np.mean(values)) if values else None
                for sigma, values in perturb_jaccards.items()
            },
            "native_top5_jaccard_under_measurement_noise": {
                str(sigma): float(np.mean(values)) if values else None
                for sigma, values in native_perturb_jaccards.items()
            },
            "explanation_latency_ms": explanation_ms,
        }
        results.append(result)
    return results


def split_normal_rows(split: Data4CyberSplit, fraction: float) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    cut = int(len(split.attack) * fraction)
    if cut < 2 or cut >= len(split.attack):
        raise ValueError(f"Normal split too short in {split.scenario}")
    train = split.features.iloc[:cut].copy()
    cal = split.features.iloc[cut:].copy()
    return train, cal, split.timestamps[:cut], split.timestamps[cut:]


def exact_sign_flip_p(differences: list[float]) -> float | None:
    diffs = np.asarray(differences, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) == 0:
        return None
    observed = abs(float(diffs.mean()))
    values = []
    for mask in range(1 << len(diffs)):
        signs = np.array([1.0 if (mask >> j) & 1 else -1.0 for j in range(len(diffs))])
        values.append(abs(float(np.mean(signs * diffs))))
    return sum(value >= observed - 1e-12 for value in values) / len(values)


def verify_split_inputs(data_dir: Path, scenario_folders: list[str]) -> dict[str, object]:
    """Hash every split CSV consumed by this run against the release manifest."""
    manifest_path = data_dir / "SHA256SUMS.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing embedded SHA-256 manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verified: dict[str, str] = {}
    for scenario in scenario_folders:
        for filename in ("dataset_attack.csv", "dataset_benign.csv"):
            relative = f"{scenario}/{filename}"
            path = data_dir / scenario / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            expected = manifest.get(relative)
            if not expected:
                raise ValueError(f"Input absent from release manifest: {relative}")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest.lower() != str(expected).lower():
                raise ValueError(f"Input SHA-256 mismatch for {relative}: {digest}")
            verified[relative] = digest
    return {"manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "verified_split_files": verified, "verified_split_file_count": len(verified)}


def holm(p_values: list[float | None]) -> list[float | None]:
    valid = [(index, value) for index, value in enumerate(p_values) if value is not None]
    ordered = sorted(valid, key=lambda pair: pair[1])
    adjusted: list[float | None] = [None] * len(p_values)
    running = 0.0
    total = len(ordered)
    for rank, (index, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * value))
        adjusted[index] = running
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser(description="Scenario-family-held-out Data4Cyber unsupervised detector audit")
    parser.add_argument("--config", type=Path, default=Path("configs/data4cyber_v3.json"))
    parser.add_argument("--output", type=Path, default=Path("results/summaries/data4cyber_loco_baselines.json"))
    parser.add_argument("--data-dir", type=Path, help="Override config data_dir, e.g. for the schema-audit subset")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    data_dir = args.data_dir if args.data_dir is not None else Path(config["data_dir"])
    if not data_dir.is_absolute():
        data_dir = args.config.resolve().parent.parent / data_dir
    config["data_dir"] = str(data_dir)
    audit_path = Path(config.get("metadata_audit_report", "results/data4cyber_metadata_audit.json"))
    if not audit_path.is_absolute():
        audit_path = args.config.resolve().parent.parent / audit_path
    metadata_audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.is_file() else None
    families: dict[str, list[str]] = config["attack_family_folds"]
    scenario_folders = sorted({folder for group in families.values() for folder in group} | {"S0_benign_baseline"})
    dataset_integrity = verify_split_inputs(data_dir, scenario_folders)
    fraction = float(config["train_fraction_within_normal_scenarios"])
    targets = [float(value) for value in config["false_positive_rate_targets"]]
    target_path = Path(config["response_target_annotations"])
    if not target_path.is_absolute():
        target_path = args.config.resolve().parent.parent / target_path
    target_annotations = pd.read_csv(target_path)

    attack_splits: dict[str, tuple[Data4CyberSplit, Data4CyberSplit]] = {}
    for folder_names in families.values():
        for folder in folder_names:
            attack_splits[folder] = load_data4cyber_scenario(data_dir / folder)
    baseline = read_benign(data_dir / config["normal_only_baseline"])
    all_normal: dict[str, Data4CyberSplit] = {"S0_benign_baseline": baseline}
    for folder, (_, benign) in attack_splits.items():
        all_normal[folder] = benign

    records: list[dict[str, object]] = []
    elapsed_start = time.perf_counter()
    for heldout_family, heldout_folders in families.items():
        training_folders = [name for name in all_normal if name == "S0_benign_baseline"
                            or not any(name in folders for folders in (families[heldout_family],))]
        # The held-out family, including the S1-alt repeat, is never used for fit or calibration.
        fit_parts: list[pd.DataFrame] = []
        cal_parts: list[pd.DataFrame] = []
        fit_times: list[np.ndarray] = []
        cal_times: list[np.ndarray] = []
        for folder in training_folders:
            split = all_normal[folder]
            fit_frame, cal_frame, fit_ts, cal_ts = split_normal_rows(split, fraction)
            fit_parts.append(fit_frame)
            cal_parts.append(cal_frame)
            fit_times.append(fit_ts)
            cal_times.append(cal_ts)
        training_features = pd.concat(fit_parts, ignore_index=True)
        imputer = fit_normal_median_imputer(training_features)
        x_train_parts = [imputer.transform(frame) for frame in fit_parts]
        dx_train_parts = [sequence_delta(x, ts) for x, ts in zip(x_train_parts, fit_times)]
        x_train = np.concatenate(x_train_parts, axis=0)
        dx_train = np.concatenate(dx_train_parts, axis=0)
        x_cal_parts = [imputer.transform(frame) for frame in cal_parts]
        dx_cal_parts = [sequence_delta(x, ts) for x, ts in zip(x_cal_parts, cal_times)]
        models = [RobustZDetector(), PCAResidualDetector(
            variance=float(config["pca_variance"]), max_components=int(config["pca_max_components"]))]
        for model in models:
            fit_started = time.perf_counter()
            model.fit(x_train, dx_train)
            fit_seconds = time.perf_counter() - fit_started
            cal_scores = [model.score(x, dx) for x, dx in zip(x_cal_parts, dx_cal_parts)]
            calibration_scores = np.concatenate(cal_scores)
            thresholds = {target: threshold_at_fpr(calibration_scores, target) for target in targets}
            for folder in heldout_folders:
                attack, benign = attack_splits[folder]
                test_frame, y_test, ts_test, phase_test, phase_all_test = combine_test(attack, benign)
                x_test = imputer.transform(test_frame)
                dx_test = sequence_delta(x_test, ts_test)
                score_started = time.perf_counter()
                test_scores = model.score(x_test, dx_test)
                scoring_seconds = time.perf_counter() - score_started
                sample_indices = np.linspace(0, len(x_test) - 1, min(200, len(x_test)), dtype=int)
                for sample_index in sample_indices[: min(10, len(sample_indices))]:
                    model.score(x_test[sample_index:sample_index + 1], dx_test[sample_index:sample_index + 1])
                sample_latencies = []
                for sample_index in sample_indices:
                    sample_start = time.perf_counter()
                    model.score(x_test[sample_index:sample_index + 1], dx_test[sample_index:sample_index + 1])
                    sample_latencies.append((time.perf_counter() - sample_start) * 1000.0)
                annotation_rows = target_annotations[target_annotations["scenario_folder"] == folder]
                if annotation_rows.empty:
                    raise ValueError(f"No frozen response-target annotation for {folder}")
                target_groups = sorted(annotation_rows["target_group"].astype(str).unique().tolist())
                evidence_tier = str(annotation_rows["evidence_tier"].iloc[0])
                target_status = str(annotation_rows["target_status"].iloc[0])
                for target, threshold in thresholds.items():
                    point = binary_metrics(y_test, test_scores, threshold)
                    events = event_metrics(y_test, test_scores > threshold, ts_test)
                    explanations = explanation_for_attack_events(
                        model, test_scores, threshold, x_test, dx_test, y_test, ts_test,
                        phase_test, phase_all_test, imputer.feature_names, target_groups,
                        evidence_tier, target_status, int(config["seed"]))
                    records.append({
                        "heldout_family": heldout_family,
                        "heldout_scenario": folder,
                        "training_scenarios": training_folders,
                        "model": model.name,
                        "normal_fpr_target": target,
                        "calibration_observed_fpr": float(np.mean(calibration_scores > threshold)),
                        "threshold": threshold,
                        "n_features": len(imputer.feature_names),
                        "test_rows": len(y_test),
                        "attack_rows": int(y_test.sum()),
                        "point_metrics": point,
                        "event_metrics": events,
                        "explanation_events": explanations,
                        "fit_seconds": fit_seconds,
                        "score_seconds": scoring_seconds,
                        "batch_amortized_score_us_per_row": scoring_seconds * 1e6 / max(1, len(y_test)),
                        "single_row_score_latency_p50_ms": float(np.quantile(sample_latencies, .5)),
                        "single_row_score_latency_p95_ms": float(np.quantile(sample_latencies, .95)),
                    })

    comparisons = []
    for target in targets:
        for metric in ("average_precision", "f1", "recall"):
            diffs = []
            family_rows = []
            for family in families:
                model_values: dict[str, list[float]] = {"RobustZ": [], "PCA-95": []}
                for record in records:
                    if record["heldout_family"] != family or record["normal_fpr_target"] != target:
                        continue
                    model_values[record["model"]].append(float(record["point_metrics"][metric]))
                if model_values["RobustZ"] and model_values["PCA-95"]:
                    diff = float(np.mean(model_values["PCA-95"]) - np.mean(model_values["RobustZ"]))
                    diffs.append(diff)
                    family_rows.append({"family": family, "pca_minus_robustz": diff})
            comparisons.append({"normal_fpr_target": target, "metric": metric,
                                "n_families": len(diffs), "mean_paired_difference": float(np.mean(diffs)) if diffs else None,
                                "exact_two_sided_sign_flip_p": exact_sign_flip_p(diffs),
                                "family_differences": family_rows})
    corrected = holm([row["exact_two_sided_sign_flip_p"] for row in comparisons])
    for row, adjusted in zip(comparisons, corrected):
        row["holm_adjusted_p_within_6_comparisons"] = adjusted

    explanation_summary = []
    groups_for_explanations = sorted({
        (record["heldout_family"], record["model"], record["normal_fpr_target"],
         event["evidence_tier"])
        for record in records for event in record["explanation_events"]
        if event["target_evaluation_eligible"]
    })
    for family, model_name, target, tier in groups_for_explanations:
        eligible = [event for record in records
                    if record["heldout_family"] == family and record["model"] == model_name
                    and record["normal_fpr_target"] == target
                    for event in record["explanation_events"]
                    if event["target_evaluation_eligible"] and event["evidence_tier"] == tier]
        detected = [event for event in eligible if event["detected"]]
        summary = {"heldout_family": family, "model": model_name,
                   "normal_fpr_target": target, "evidence_tier": tier,
                   "eligible_attack_events": len(eligible),
                   "detector_misses_counted_as_target_misses": sum(not event["detected"] for event in eligible)}
        for k in (1, 3, 5):
            summary[f"unconditional_hit_at_{k}"] = float(np.mean([event[f"target_hit_at_{k}"] for event in eligible])) if eligible else None
            summary[f"native_unconditional_hit_at_{k}"] = float(np.mean([event[f"native_target_hit_at_{k}"] for event in eligible])) if eligible else None
            summary[f"random_reference_unconditional_hit_probability_at_{k}"] = float(np.mean([event["random_reference_unconditional_hit_probabilities"][str(k)] for event in eligible])) if eligible else None
        summary["mean_top_group_score_reduction_when_detected"] = (
            float(np.mean([event["top_group_score_reduction"] for event in detected
                           if event["top_group_score_reduction"] is not None])) if detected else None)
        for sigma in ("0.01", "0.05"):
            for method, key in (("ablation", "top5_jaccard_under_measurement_noise"),
                                ("native", "native_top5_jaccard_under_measurement_noise")):
                values = [event[key][sigma] for event in detected
                          if event[key] is not None and event[key][sigma] is not None]
                summary[f"mean_{method}_top5_jaccard_sigma_{sigma}"] = float(np.mean(values)) if values else None
        latencies = [event["explanation_latency_ms"] for event in detected if event["explanation_latency_ms"] is not None]
        summary["explanation_latency_p50_ms"] = float(np.quantile(latencies, .5)) if latencies else None
        summary["explanation_latency_p95_ms"] = float(np.quantile(latencies, .95)) if latencies else None
        score_p95 = [record["single_row_score_latency_p95_ms"] for record in records
                     if record["heldout_family"] == family and record["model"] == model_name
                     and record["normal_fpr_target"] == target]
        summary["single_row_detector_score_latency_p95_ms"] = float(np.mean(score_p95)) if score_p95 else None
        explanation_summary.append(summary)

    output = {
        "study": "Data4Cyber v3 scenario-family-held-out process-only detector audit",
        "status": "exploratory detector and first-alarm explanation benchmark",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {"python": sys.version, "platform": platform.platform(),
                    "processor": platform.processor(), "numpy": np.__version__, "pandas": pd.__version__},
        "config": config,
        "dataset_integrity": dataset_integrity,
        "archive_aggregate": metadata_audit.get("archive_aggregate") if metadata_audit else None,
        "zenodo_record_metadata_discrepancy": metadata_audit.get("zenodo_record_metadata_as_seen_2026_10_31") if metadata_audit else None,
        "primary_attack_families": len(families),
        "heldout_family_results": records,
        "first_alarm_explanation_summary": explanation_summary,
        "paired_family_comparisons": comparisons,
        "wall_seconds": time.perf_counter() - elapsed_start,
        "limitations": [
            "One testbed and six attack-family units; statistical tests are low-powered and exploratory.",
            "S1-alt is grouped with S1 to prevent same-family contamination.",
            "This is process-only and does not reproduce the released IPAL baselines.",
            "Row-level metrics do not establish cross-site generalization.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    summary = {}
    for record in records:
        key = (record["heldout_family"], record["model"], record["normal_fpr_target"])
        summary.setdefault(key, []).append(record["point_metrics"])
    print(f"Wrote {len(records)} scenario/model/threshold records to {args.output}")
    for key, values in sorted(summary.items()):
        print(key, {metric: round(float(np.mean([value[metric] for value in values])), 4)
                    for metric in ("average_precision", "f1", "recall", "false_positive_rate")})
    print("paired comparisons", json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
