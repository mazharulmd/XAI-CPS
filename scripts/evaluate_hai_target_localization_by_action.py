from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAP = ROOT / "data" / "annotations" / "hai21_event_target_map.csv"
DEFAULT_ACTIONS = ROOT / "data" / "annotations" / "hai21_scenario_action_types.csv"
DEFAULT_RUNS = [
    ROOT / "results/runs/hai21_seed42_20261031T051335_879603Z",
    ROOT / "results/runs/hai21_seed43_20261031T051709_959445Z",
    ROOT / "results/runs/hai21_seed44_20261031T052039_198005Z",
    ROOT / "results/runs/hai21_seed45_20261031T052411_161777Z",
    ROOT / "results/runs/hai21_seed46_20261031T052743_620105Z",
]


def split_tokens(value: object, separator: str = "|") -> set[str]:
    if value is None or pd.isna(value):
        return set()
    return {token.strip() for token in str(value).split(separator) if token.strip()}


def action_class(scenarios: set[str], action_types: dict[str, set[str]]) -> str:
    missing = scenarios - action_types.keys()
    if missing:
        raise ValueError(f"No official action-type annotation for scenarios: {sorted(missing)}")
    types = set().union(*(action_types[s] for s in scenarios)) if scenarios else set()
    return "+".join(sorted(types)) if types else "UNMAPPED"


def hit(ranking: object, targets: set[str], k: int) -> float:
    ranked = str(ranking).split("|") if ranking is not None and not pd.isna(ranking) else []
    return float(bool(set(ranked[:k]) & targets))


def _cf_hit(row: dict, targets: set[str], success_key: str, tags_key: str) -> float:
    if row.get(success_key) is None or pd.isna(row.get(success_key)) or not bool(row[success_key]):
        return 0.0
    return float(bool(targets & split_tokens(row.get(tags_key), separator=";")))


def evaluate(runs: list[Path], mapping_path: Path, actions_path: Path) -> dict:
    mapping = pd.read_csv(mapping_path, keep_default_na=False)
    actions = pd.read_csv(actions_path, keep_default_na=False)
    action_types = {
        row.scenario_id: split_tokens(row.variable_types)
        for row in actions.itertuples(index=False)
    }
    mapping["scenario_set"] = mapping["scenario_ids"].map(split_tokens)
    mapping["action_class"] = mapping["scenario_set"].map(lambda items: action_class(items, action_types))
    mapping["target_annotation_complete"] = mapping["target_annotation_complete"].astype(str).str.lower().eq("true")
    eligible = mapping.loc[mapping["target_annotation_complete"]].copy()
    eligible["event_key"] = eligible["attack_id"].astype(str)

    run_rows = []
    models: set[str] = set()
    for run_dir in runs:
        if not (run_dir / "event_explanations.csv").is_file():
            raise FileNotFoundError(f"Missing event explanations: {run_dir / 'event_explanations.csv'}")
        frame = pd.read_csv(run_dir / "event_explanations.csv")
        frame["event_index"] = frame["event_index"].astype(int)
        models.update(frame["model"].astype(str).unique())
        frame["seed"] = int(run_dir.name.split("seed", 1)[1].split("_", 1)[0])
        run_rows.append(frame)
    all_rows = pd.concat(run_rows, ignore_index=True)

    event_outcomes: list[dict] = []
    for event in eligible.to_dict("records"):
        targets = split_tokens(event["target_points_present_in_csv"])
        key = (event["dataset_file"], int(event["event_index"]))
        for model in sorted(models):
            rows = all_rows.loc[
                (all_rows["file"] == key[0])
                & (all_rows["event_index"] == key[1])
                & (all_rows["model"] == model)
            ]
            for seed in sorted(int(r.name.split("seed", 1)[1].split("_", 1)[0]) for r in runs):
                match = rows.loc[rows["seed"] == seed]
                if len(match) > 1:
                    raise ValueError(f"Duplicate event explanation for {event['attack_id']}, {model}, seed {seed}")
                row = match.iloc[0].to_dict() if len(match) else {}
                detected = bool(row) and pd.notna(row.get("native_top10_tags"))
                outcome = {
                    "attack_id": event["attack_id"],
                    "dataset_file": event["dataset_file"],
                    "seed": seed,
                    "model": model,
                    "action_class": event["action_class"],
                    "scenario_ids": event["scenario_ids"],
                    "targets": sorted(targets),
                    "detected": bool(detected),
                }
                for k in (1, 3, 5):
                    outcome[f"native_hit_at_{k}"] = hit(row.get("native_top10_tags"), targets, k)
                    outcome[f"grouped_ablation_hit_at_{k}"] = hit(row.get("ablation_top10_tags"), targets, k)
                outcome["support_cf_target_hit"] = _cf_hit(
                    row, targets, "capability_support_success", "capability_edit_names"
                )
                outcome["screened_cf_target_hit"] = _cf_hit(
                    row, targets, "dynamics_constrained_success", "dynamics_constrained_edit_names"
                )
                outcome["support_cf_success"] = bool(row.get("capability_support_success", False)) if row else False
                outcome["screened_cf_success"] = bool(row.get("dynamics_constrained_success", False)) if row else False
                event_outcomes.append(outcome)

    outcomes = pd.DataFrame(event_outcomes)
    summary = {}
    metric_names = [
        "detected", "native_hit_at_1", "native_hit_at_3", "native_hit_at_5",
        "grouped_ablation_hit_at_1", "grouped_ablation_hit_at_3", "grouped_ablation_hit_at_5",
        "support_cf_target_hit", "screened_cf_target_hit", "support_cf_success", "screened_cf_success",
    ]
    for (model, action), group in outcomes.groupby(["model", "action_class"], sort=True):
        # Seeds repeat the same scenarios. Average within attack ID before averaging
        # across IDs so seeds never inflate the descriptive scenario denominator.
        per_attack = group.groupby("attack_id", sort=True)[metric_names].mean()
        summary.setdefault(model, {})[action] = {
            "n_attack_ids": int(per_attack.shape[0]),
            "attack_ids": sorted(per_attack.index.astype(str).tolist()),
            "metrics_mean_across_attack_ids": {
                metric: float(per_attack[metric].mean()) for metric in metric_names
            },
            "support_cf_target_hit_given_success": (
                float(group.loc[group["support_cf_success"], "support_cf_target_hit"].mean())
                if group["support_cf_success"].any() else None
            ),
            "screened_cf_target_hit_given_success": (
                float(group.loc[group["screened_cf_success"], "screened_cf_target_hit"].mean())
                if group["screened_cf_success"].any() else None
            ),
        }
    return {
        "dataset": "HAI 21.03",
        "action_type_source": "Shin et al., CSET 2021, Appendix A; scenario types transcribed to hai21_scenario_action_types.csv",
        "target_ground_truth_source": "HAI maintainer technical manual, aligned to CSV attack intervals",
        "eligible_exact_target_events": int(eligible["attack_id"].nunique()),
        "ineligible_events_due_to_incomplete_exact_targets": int(mapping["attack_id"].nunique() - eligible["attack_id"].nunique()),
        "capability_labels_used_for_explanation_generation": False,
        "seeds_are_repeated_measurements_of_the_same_attack_ids": True,
        "interpretation": (
            "Descriptive, action-type-stratified target localization only. Scenario action labels are post-hoc evaluation strata; "
            "they are not passed to detectors, explainers, or candidate searches. Missed detections and failed counterfactuals count as misses. "
            "No subgroup significance tests are reported because action-type strata are small and repeated seeds reuse the same attack IDs. "
            "Documented target agreement is not causal validation or proof of physical feasibility."
        ),
        "summary": summary,
        "event_outcomes": event_outcomes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stratify exact HAI target localization by documented attack action type")
    parser.add_argument("runs", nargs="*", type=Path, default=DEFAULT_RUNS, help="identity-consistent HAI run directories")
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--actions", type=Path, default=DEFAULT_ACTIONS)
    parser.add_argument("--output", type=Path, default=ROOT / "results/summaries/hai21_identity_consistent_target_localization_by_action_seed42_46.json")
    args = parser.parse_args()
    result = evaluate(args.runs, args.mapping, args.actions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for model, strata in result["summary"].items():
        print(model)
        for action, values in strata.items():
            metrics = values["metrics_mean_across_attack_ids"]
            print(
                f"  {action}: n={values['n_attack_ids']} detected={metrics['detected']:.3f} "
                f"native@5={metrics['native_hit_at_5']:.3f} ablation@5={metrics['grouped_ablation_hit_at_5']:.3f} "
                f"supportCF={metrics['support_cf_target_hit']:.3f} screenedCF={metrics['screened_cf_target_hit']:.3f}"
            )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
