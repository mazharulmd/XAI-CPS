from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import numpy as np
import pandas as pd

from xai_cps.target_eval import random_hit_probability

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAP = ROOT / "data" / "annotations" / "hai21_event_target_map.csv"
HAI_NUMERIC_TAG_COUNT = 79


def _tokens(value: object, separator: str = "|") -> set[str]:
    if value is None or pd.isna(value):
        return set()
    return {part for part in str(value).split(separator) if part}


def _successful_target_hit(row: dict, targets: set[str], success_column: str, edits_column: str) -> float:
    """Return an exact-target hit, treating missing detections/failed CFs as misses."""
    success = row.get(success_column)
    if pd.isna(success) or not bool(success):
        return 0.0
    return float(bool(targets & _tokens(row.get(edits_column), separator=";")))


def evaluate_run(run_dir: Path, mapping: pd.DataFrame) -> dict:
    explanations = pd.read_csv(run_dir / "event_explanations.csv")
    explanations["event_index"] = explanations["event_index"].astype(int)
    eligible = mapping.loc[mapping["target_annotation_complete"].astype(bool)].copy()
    expected_by_model = {}
    outcomes_by_model = {}
    for model, model_rows in explanations.groupby("model"):
        all_rows = eligible.merge(
            model_rows,
            left_on=["dataset_file", "event_index"],
            right_on=["file", "event_index"],
            how="left",
            validate="one_to_one",
        )
        rows = all_rows.loc[all_rows["native_top10_tags"].notna()]
        eligible_count = int(len(eligible))
        detected_count = int(len(rows))
        metric_values = {
            "native_hit_at_1": [],
            "native_hit_at_3": [],
            "native_hit_at_5": [],
            "grouped_ablation_hit_at_1": [],
            "grouped_ablation_hit_at_3": [],
            "grouped_ablation_hit_at_5": [],
            "capability_cf_any_target_hit_given_success": [],
            "dynamics_cf_any_target_hit_given_success": [],
        }
        for row in rows.to_dict("records"):
            targets = _tokens(row["target_points_present_in_csv"])
            native = str(row["native_top10_tags"]).split("|")
            ablation = str(row["ablation_top10_tags"]).split("|")
            for k in (1, 3, 5):
                metric_values[f"native_hit_at_{k}"].append(float(bool(targets & set(native[:k]))))
                metric_values[f"grouped_ablation_hit_at_{k}"].append(float(bool(targets & set(ablation[:k]))))
            if bool(row["capability_support_success"]):
                edits = _tokens(row["capability_edit_names"], separator=";")
                metric_values["capability_cf_any_target_hit_given_success"].append(
                    float(bool(targets & edits))
                )
            if pd.notna(row.get("dynamics_constrained_success")) and bool(row["dynamics_constrained_success"]):
                edits = _tokens(row["dynamics_constrained_edit_names"], separator=";")
                metric_values["dynamics_cf_any_target_hit_given_success"].append(float(bool(targets & edits)))
        cf_successes = len(metric_values["capability_cf_any_target_hit_given_success"])
        dynamics_cf_successes = len(metric_values["dynamics_cf_any_target_hit_given_success"])
        event_outcomes = []
        for row in all_rows.to_dict("records"):
            detected = pd.notna(row.get("native_top10_tags"))
            targets = _tokens(row["target_points_present_in_csv"])
            native = str(row["native_top10_tags"]).split("|") if detected else []
            ablation = str(row["ablation_top10_tags"]).split("|") if detected else []
            event_result = {
                "attack_id": row["attack_id"],
                "dataset_file": row["dataset_file"],
                "detected_and_explained": bool(detected),
                "dynamics_constrained_success": (
                    bool(row["dynamics_constrained_success"])
                    if pd.notna(row.get("dynamics_constrained_success")) else False
                ),
                # Unconditional over every exact-target-complete event: left-join
                # misses and unsuccessful/empty counterfactuals are all zero.
                "capability_cf_target_hit_unconditional": _successful_target_hit(
                    row, targets, "capability_support_success", "capability_edit_names"
                ),
                "dynamics_cf_target_hit_unconditional": _successful_target_hit(
                    row, targets, "dynamics_constrained_success", "dynamics_constrained_edit_names"
                ),
            }
            for k in (1, 3, 5):
                event_result[f"native_hit_at_{k}"] = float(bool(targets & set(native[:k])))
                event_result[f"grouped_ablation_hit_at_{k}"] = float(bool(targets & set(ablation[:k])))
                event_result[f"random_reference_hit_probability_at_{k}"] = (
                    random_hit_probability(len(targets), HAI_NUMERIC_TAG_COUNT, k) if detected else 0.0
                )
            event_outcomes.append(event_result)
        per_run = {
            "eligible_target_complete_events": eligible_count,
            "detected_and_explained_eligible_events": detected_count,
            "detected_fraction_of_eligible_events": detected_count / max(1, eligible_count),
            "capability_counterfactual_successes_on_eligible_events": cf_successes,
            "dynamics_constrained_counterfactual_successes_on_eligible_events": dynamics_cf_successes,
            "capability_cf_target_hits_unconditional": int(sum(
                row["capability_cf_target_hit_unconditional"] for row in event_outcomes
            )),
            "dynamics_cf_target_hits_unconditional": int(sum(
                row["dynamics_cf_target_hit_unconditional"] for row in event_outcomes
            )),
            "capability_cf_target_hit_rate_unconditional": float(np.mean([
                row["capability_cf_target_hit_unconditional"] for row in event_outcomes
            ])) if event_outcomes else None,
            "dynamics_cf_target_hit_rate_unconditional": float(np.mean([
                row["dynamics_cf_target_hit_unconditional"] for row in event_outcomes
            ])) if event_outcomes else None,
            "random_reference_unconditional_hit_at_k": {
                str(k): float(np.mean([
                    row[f"random_reference_hit_probability_at_{k}"] for row in event_outcomes
                ])) if event_outcomes else None
                for k in (1, 3, 5)
            },
            "metrics_conditional_on_detected_event": {
                name: (statistics.mean(values) if values else None)
                for name, values in metric_values.items()
            },
        }
        expected_by_model[model] = per_run
        outcomes_by_model[model] = event_outcomes
    return {
        "run_id": run_dir.name,
        "seed": int(run_dir.name.split("seed", 1)[1].split("_", 1)[0]),
        "models": expected_by_model,
        "event_outcomes": outcomes_by_model,
    }


def paired_localization_tests(runs: list[dict], seed: int = 271828, draws: int = 5000) -> dict:
    """Exploratory paired tests; attack IDs are resampled, with files as clusters."""
    rng = np.random.default_rng(seed)
    result = {}
    for model in sorted(runs[0]["event_outcomes"]):
        pooled: dict[str, list[dict]] = {}
        for run in runs:
            for row in run["event_outcomes"][model]:
                pooled.setdefault(row["attack_id"], []).append(row)
        attacks = sorted(pooled)
        files = {attack: pooled[attack][0]["dataset_file"] for attack in attacks}
        comparisons = {}
        for k in (1, 3, 5):
            differences = np.asarray(
                [
                    np.mean([x[f"grouped_ablation_hit_at_{k}"] - x[f"native_hit_at_{k}"] for x in pooled[attack]])
                    for attack in attacks
                ],
                dtype=float,
            )
            observed = float(differences.mean())
            event_boot = np.empty(draws, dtype=float)
            for b in range(draws):
                event_boot[b] = float(np.mean(differences[rng.integers(0, len(attacks), len(attacks))]))
            file_names = sorted(set(files.values()))
            file_boot = np.empty(draws, dtype=float)
            indices_by_file = {name: np.flatnonzero([files[a] == name for a in attacks]) for name in file_names}
            for b in range(draws):
                sampled_files = rng.choice(file_names, size=len(file_names), replace=True)
                sampled_indices = np.concatenate([indices_by_file[name] for name in sampled_files])
                file_boot[b] = float(np.mean(differences[sampled_indices]))
            perm = np.empty(draws, dtype=float)
            for b in range(draws):
                signs = rng.choice(np.array([-1.0, 1.0]), size=len(attacks))
                perm[b] = float(np.mean(differences * signs))
            p_value = (int(np.sum(np.abs(perm) >= abs(observed))) + 1) / (draws + 1)
            comparisons[str(k)] = {
                "paired_difference_grouped_ablation_minus_native_hit_rate": observed,
                "attack_event_bootstrap_ci95": [float(x) for x in np.quantile(event_boot, [0.025, 0.975])],
                "file_cluster_bootstrap_ci95": [float(x) for x in np.quantile(file_boot, [0.025, 0.975])],
                "paired_sign_flip_p_two_sided": float(p_value),
                "n_attack_events": len(attacks),
                "n_file_clusters": len(file_names),
                "bootstrap_draws": draws,
                "estimand": "unconditional exact-target hit rate; missed detections count as no explanation hit",
            }
        result[model] = comparisons
    return result


def paired_counterfactual_localization_tests(
    runs: list[dict], seed: int = 314159, draws: int = 10000
) -> dict:
    """Paired event-level comparison of transition-screened vs support-only CF hits.

    Seeds are averaged within attack ID first. The estimand is the change in the
    probability of an exact documented-target intersection over all eligible
    attack events; missed detections and failed CFs count as misses.
    """
    if not runs:
        return {}
    rng = np.random.default_rng(seed)
    result: dict[str, dict] = {}
    raw_p: dict[str, float] = {}
    for model in sorted(runs[0]["event_outcomes"]):
        pooled: dict[str, list[dict]] = {}
        for run in runs:
            for row in run["event_outcomes"][model]:
                pooled.setdefault(row["attack_id"], []).append(row)
        attacks = sorted(pooled)
        differences = np.asarray([
            np.mean([
                row["dynamics_cf_target_hit_unconditional"]
                - row["capability_cf_target_hit_unconditional"]
                for row in pooled[attack]
            ])
            for attack in attacks
        ], dtype=float)
        observed = float(differences.mean()) if len(differences) else 0.0
        boot = np.empty(draws, dtype=float)
        for b in range(draws):
            boot[b] = float(np.mean(differences[rng.integers(0, len(attacks), len(attacks))]))
        perm = np.empty(draws, dtype=float)
        for b in range(draws):
            perm[b] = float(np.mean(differences * rng.choice(np.array([-1.0, 1.0]), size=len(attacks))))
        p_value = (int(np.sum(np.abs(perm) >= abs(observed))) + 1) / (draws + 1)
        raw_p[model] = float(p_value)
        files = sorted({pooled[a][0]["dataset_file"] for a in attacks})
        result[model] = {
            "paired_difference_dynamics_minus_capability_unconditional_target_hit_rate": observed,
            "attack_event_bootstrap_ci95": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
            "paired_sign_flip_p_two_sided": float(p_value),
            "n_attack_events": len(attacks),
            "n_file_clusters": len(files),
            "bootstrap_draws": draws,
            "estimand": "all exact-target-complete events; missed detections and unsuccessful counterfactuals count as no target hit",
        }
    # Holm step-down adjustment across the three model-level comparisons.
    ordered = sorted(raw_p, key=raw_p.get)
    adjusted = {}
    running = 0.0
    m = len(ordered)
    for rank, model in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * raw_p[model]))
        adjusted[model] = float(running)
    for model in result:
        result[model]["holm_adjusted_p_two_sided"] = adjusted[model]
    return result


def random_ranking_benchmark_tests(
    runs: list[dict], seed: int = 161803, draws: int = 20000
) -> dict:
    """Compare observed first-alarm Hit@k with an event-matched random-ranking null.

    The target set and detector-hit/miss pattern are held fixed. For every
    detected event and seed, the null Hit@k is Bernoulli with its exact
    hypergeometric probability under a uniformly sampled size-k ranking of the
    79 numeric tags. Seed outcomes are averaged within attack ID before the
    event-level estimate and bootstrap are computed.
    """
    if draws < 1000:
        raise ValueError("draws must be at least 1000 for the Monte Carlo null")
    if not runs:
        return {}
    rng = np.random.default_rng(seed)
    result: dict[str, dict] = {}
    raw_p: dict[str, float] = {}
    for model in sorted(runs[0]["event_outcomes"]):
        pooled: dict[str, list[dict]] = {}
        for run in runs:
            for row in run["event_outcomes"][model]:
                pooled.setdefault(row["attack_id"], []).append(row)
        attacks = sorted(pooled)
        files = {attack: pooled[attack][0]["dataset_file"] for attack in attacks}
        model_result: dict[str, dict] = {}
        for method in ("native", "grouped_ablation"):
            for k in (1, 3, 5):
                observed_by_event = np.asarray([
                    np.mean([row[f"{method}_hit_at_{k}"] for row in pooled[attack]])
                    for attack in attacks
                ], dtype=float)
                probability_by_event = np.asarray([
                    np.mean([row[f"random_reference_hit_probability_at_{k}"] for row in pooled[attack]])
                    for attack in attacks
                ], dtype=float)
                paired_excess = observed_by_event - probability_by_event
                observed_excess = float(paired_excess.mean())
                boot = np.empty(draws, dtype=float)
                for b in range(draws):
                    boot[b] = float(np.mean(
                        paired_excess[rng.integers(0, len(attacks), len(attacks))]
                    ))
                file_names = sorted(set(files.values()))
                indices_by_file = {
                    name: np.flatnonzero([files[attack] == name for attack in attacks])
                    for name in file_names
                }
                file_boot = np.empty(draws, dtype=float)
                for b in range(draws):
                    sampled_files = rng.choice(file_names, size=len(file_names), replace=True)
                    sampled_indices = np.concatenate([indices_by_file[name] for name in sampled_files])
                    file_boot[b] = float(np.mean(paired_excess[sampled_indices]))

                # Event/seed random rankings are sampled conditionally on the
                # observed detector hits; missed events have null probability 0.
                event_seed_probabilities = np.asarray([
                    [row[f"random_reference_hit_probability_at_{k}"] for row in pooled[attack]]
                    for attack in attacks
                ], dtype=float)
                null_hits = rng.random((draws, *event_seed_probabilities.shape)) < event_seed_probabilities[None, :, :]
                null_estimates = null_hits.mean(axis=2).mean(axis=1)
                null_center = float(probability_by_event.mean())
                p_value = float((np.count_nonzero(
                    np.abs(null_estimates - null_center) >= abs(observed_excess)
                ) + 1) / (draws + 1))
                key = f"{model}|{method}|{k}"
                raw_p[key] = p_value
                model_result.setdefault(method, {})[str(k)] = {
                    "observed_unconditional_hit_rate": float(observed_by_event.mean()),
                    "random_reference_unconditional_hit_rate": null_center,
                    "excess_hit_rate_over_random_reference": observed_excess,
                    "event_cluster_bootstrap_ci95_for_excess": [
                        float(x) for x in np.quantile(boot, [0.025, 0.975])
                    ],
                    "file_cluster_bootstrap_ci95_for_excess": [
                        float(x) for x in np.quantile(file_boot, [0.025, 0.975])
                    ],
                    "conditional_random_ranking_p_two_sided": p_value,
                    "n_attack_event_clusters": len(attacks),
                    "n_test_file_clusters": len(set(files.values())),
                    "monte_carlo_draws": draws,
                    "estimand": (
                        "exact-target-complete HAI events; repeated seeds averaged within attack ID; "
                        "detector misses fixed at zero under both observed and random rankings"
                    ),
                }
        result[model] = model_result

    # One Holm family covers all model, explainer, and K comparisons.
    ordered = sorted(raw_p, key=raw_p.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, key in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * raw_p[key]))
        adjusted[key] = float(running)
    for model in result:
        for method in result[model]:
            for k in result[model][method]:
                key = f"{model}|{method}|{k}"
                result[model][method][k]["holm_adjusted_p_two_sided"] = adjusted[key]
    return result


def aggregate(runs: list[dict]) -> dict:
    model_names = sorted(runs[0]["models"])
    summary = {}
    for model in model_names:
        vals = [run["models"][model] for run in runs]
        metrics = sorted({
            key
            for item in vals
            for key, value in item["metrics_conditional_on_detected_event"].items()
            if value is not None
        })
        summary[model] = {
            "eligible_target_complete_events": vals[0]["eligible_target_complete_events"],
            "detected_and_explained_eligible_events": {
                "mean": statistics.mean(item["detected_and_explained_eligible_events"] for item in vals),
                "range": [
                    min(item["detected_and_explained_eligible_events"] for item in vals),
                    max(item["detected_and_explained_eligible_events"] for item in vals),
                ],
            },
            "detected_fraction_of_eligible_events": {
                "mean": statistics.mean(item["detected_fraction_of_eligible_events"] for item in vals),
                "range": [
                    min(item["detected_fraction_of_eligible_events"] for item in vals),
                    max(item["detected_fraction_of_eligible_events"] for item in vals),
                ],
            },
            "random_reference_unconditional_hit_at_k": {
                str(k): {
                    "mean": statistics.mean(item["random_reference_unconditional_hit_at_k"][str(k)] for item in vals),
                    "range": [
                        min(item["random_reference_unconditional_hit_at_k"][str(k)] for item in vals),
                        max(item["random_reference_unconditional_hit_at_k"][str(k)] for item in vals),
                    ],
                }
                for k in (1, 3, 5)
            },
            "capability_counterfactual_successes_on_eligible_events": {
                "mean": statistics.mean(item["capability_counterfactual_successes_on_eligible_events"] for item in vals),
                "range": [
                    min(item["capability_counterfactual_successes_on_eligible_events"] for item in vals),
                    max(item["capability_counterfactual_successes_on_eligible_events"] for item in vals),
                ],
            },
            "target_localization": {
                key: {
                    "mean": statistics.mean(item["metrics_conditional_on_detected_event"][key] for item in vals),
                    "range": [
                        min(item["metrics_conditional_on_detected_event"][key] for item in vals),
                        max(item["metrics_conditional_on_detected_event"][key] for item in vals),
                    ],
                }
                for key in metrics
                if all(item["metrics_conditional_on_detected_event"][key] is not None for item in vals)
            },
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HAI alarm explanations against documented target tags")
    parser.add_argument("runs", nargs="+", type=Path, help="HAI run directories with event_explanations.csv")
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--randomization-draws", type=int, default=20000)
    parser.add_argument("--randomization-seed", type=int, default=161803)
    args = parser.parse_args()

    mapping = pd.read_csv(args.mapping)
    if mapping["attack_id"].duplicated().any():
        raise ValueError("Event target mapping contains duplicate attack IDs")
    if not mapping["stage_labels_match_controller_metadata"].astype(bool).all():
        raise ValueError("Event target mapping failed the documented controller-stage cross-check")
    run_results = [evaluate_run(Path(run), mapping) for run in args.runs]
    result = {
        "dataset": "HAI 21.03",
        "target_ground_truth_source": "Official HAI technical manual scenario timetable, aligned to test CSV intervals by timestamp",
        "only_exact_csv_target_points_counted_as_gold": True,
        "random_ranking_reference": {
            "group_universe_size": HAI_NUMERIC_TAG_COUNT,
            "definition": "Analytic uniform top-k without-replacement target-intersection probability; detector misses contribute zero to unconditional reference rates.",
        },
        "incomplete_target_events_excluded_from_tag_localization": int((~mapping["target_annotation_complete"].astype(bool)).sum()),
        "eligible_exact_target_events": int(mapping["target_annotation_complete"].astype(bool).sum()),
        "runs": run_results,
        "aggregate": aggregate(run_results),
        "paired_localization_tests": paired_localization_tests(run_results),
        "random_ranking_benchmark_tests": random_ranking_benchmark_tests(
            run_results, seed=args.randomization_seed, draws=args.randomization_draws
        ),
        "unconditional_counterfactual_target_localization": {
            model: {
                "eligible_target_complete_events": run_results[0]["models"][model]["eligible_target_complete_events"],
                "capability_support_only_hit_count_by_seed": [
                    run["models"][model]["capability_cf_target_hits_unconditional"] for run in run_results
                ],
                "capability_support_only_hit_rate_mean_across_seeds": statistics.mean(
                    run["models"][model]["capability_cf_target_hit_rate_unconditional"] for run in run_results
                ),
                "dynamics_constrained_hit_count_by_seed": [
                    run["models"][model]["dynamics_cf_target_hits_unconditional"] for run in run_results
                ],
                "dynamics_constrained_hit_rate_mean_across_seeds": statistics.mean(
                    run["models"][model]["dynamics_cf_target_hit_rate_unconditional"] for run in run_results
                ),
            }
            for model in sorted(run_results[0]["models"])
        },
        "paired_counterfactual_localization_tests": paired_counterfactual_localization_tests(run_results),
        "interpretation": (
            "All metrics are descriptive over supplied seeds; ranges are seed ranges, not confidence intervals. "
            "Native/grouped target hit@k is conditional on a detector producing an alarm explanation in that event. "
            "The unconditional counterfactual target-hit analysis uses every exact-target-complete event, with missed detections and unsuccessful counterfactuals counted as misses. "
            "Events with target names absent from the HAI CSV schema are excluded rather than mapped by guesswork. "
            "The result measures localization of documented target points, not causal root-cause identification."
        ),
    }
    payload = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
