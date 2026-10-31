from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from xai_cps.data import contiguous_events, load_batadal
from xai_cps.target_eval import (
    aggregate_component_scores, component_hit_at_k, pump_component_from_tag, rank_components,
)


ANNOTATION_FILE = ROOT / "data" / "annotations" / "batadal_attack_target_map.csv"
CONFIG_FILE = ROOT / "configs" / "batadal.json"


def load_target_annotations(path: Path = ANNOTATION_FILE) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["start_date_inclusive"] = date.fromisoformat(row["start_date_inclusive"])
        row["end_date_inclusive"] = date.fromisoformat(row["end_date_inclusive"])
        row["targets"] = row["target_pumps"].split("|")
    return rows


def align_annotations_to_labeled_events(segment, annotations: list[dict]) -> list[dict]:
    labels = np.asarray(segment.attack, dtype=bool) & np.asarray(segment.label_known, dtype=bool)
    events = contiguous_events(labels)
    aligned = []
    used_indices = set()
    for annotation in annotations:
        matches = []
        for event_index, (start, end) in enumerate(events):
            event_start = date.fromisoformat(str(segment.timestamps[start])[:10])
            event_end = date.fromisoformat(str(segment.timestamps[end - 1])[:10])
            if (annotation["start_date_inclusive"] <= event_start <= annotation["end_date_inclusive"]
                    and annotation["start_date_inclusive"] <= event_end <= annotation["end_date_inclusive"]):
                matches.append((event_index, start, end))
        if len(matches) != 1:
            raise ValueError(f"{annotation['event_id']} matched {len(matches)} labeled intervals; expected one")
        event_index, start, end = matches[0]
        if event_index in used_indices:
            raise ValueError(f"multiple annotations map to labeled interval index {event_index}")
        used_indices.add(event_index)
        aligned.append({
            **annotation,
            "event_index": event_index,
            "start_row": start,
            "end_row_exclusive": end,
            "observed_start": str(segment.timestamps[start]),
            "observed_end_inclusive": str(segment.timestamps[end - 1]),
            "positive_rows": end - start,
        })
    if len(used_indices) != len(events):
        raise ValueError(f"annotated {len(used_indices)} of {len(events)} approximate positive intervals")
    return aligned


def _scores_to_ranking(scores: dict | None) -> list[str]:
    if not isinstance(scores, dict):
        return []
    return rank_components(aggregate_component_scores(scores))


def evaluate_run(run_dir: Path, aligned: list[dict]) -> tuple[dict, list[dict]]:
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    per_model_events = defaultdict(dict)
    for row in summary["explanations"]:
        per_model_events[row["model"]][int(row["event_index"])] = row
    records = []
    for model, model_summary in summary["models"].items():
        flags = model_summary.get("event_detection_by_index")
        if flags is None or len(flags) != len(aligned):
            raise ValueError(f"{summary['run_id']}/{model}: missing per-event detection flags; rerun updated runner")
        for annotation in aligned:
            event_index = annotation["event_index"]
            event = per_model_events[model].get(event_index)
            detected = bool(flags[event_index])
            if detected != (event is not None):
                raise ValueError(f"{summary['run_id']}/{model}/{event_index}: detection flag and explanation row disagree")
            native_scores = event.get("native_tag_scores") if event else None
            ablation_scores = event.get("grouped_ablation_tag_scores") if event else None
            native_rank = _scores_to_ranking(native_scores)
            ablation_rank = _scores_to_ranking(ablation_scores)
            edits = (event.get("capability_edit_names", "").split(";")
                     if event and event.get("capability_edit_names") else [])
            edit_components = sorted({component for tag in edits
                                      if (component := pump_component_from_tag(tag))})
            targets = annotation["targets"]
            recourse_passed = bool(event and event.get("capability_support_scoreflip"))
            recourse_hit = bool(recourse_passed and set(edit_components) & set(targets))
            records.append({
                "run_id": summary["run_id"],
                "seed": int(summary["seed"]),
                "model": model,
                "event_id": annotation["event_id"],
                "event_index": event_index,
                "targets": targets,
                "detected": detected,
                "native_component_ranking": native_rank,
                "grouped_ablation_component_ranking": ablation_rank,
                "native_hit_at_1": component_hit_at_k(native_rank, targets, 1),
                "native_hit_at_3": component_hit_at_k(native_rank, targets, 3),
                "native_hit_at_5": component_hit_at_k(native_rank, targets, 5),
                "grouped_ablation_hit_at_1": component_hit_at_k(ablation_rank, targets, 1),
                "grouped_ablation_hit_at_3": component_hit_at_k(ablation_rank, targets, 3),
                "grouped_ablation_hit_at_5": component_hit_at_k(ablation_rank, targets, 5),
                "recourse_passed": recourse_passed,
                "recourse_edited_pumps": edit_components,
                "recourse_target_hit_unconditional": recourse_hit,
                "recourse_target_hit_if_pass": recourse_hit if recourse_passed else None,
            })
    return summary, records


def summarize(records: list[dict]) -> dict:
    by_model = defaultdict(list)
    for row in records:
        by_model[row["model"]].append(row)
    results = {}
    for model, rows in sorted(by_model.items()):
        metrics = [
            "native_hit_at_1", "native_hit_at_3", "native_hit_at_5",
            "grouped_ablation_hit_at_1", "grouped_ablation_hit_at_3", "grouped_ablation_hit_at_5",
            "recourse_target_hit_unconditional",
        ]
        results[model] = {
            "n_seed_event_rows": len(rows),
            "n_unique_events": len({row["event_id"] for row in rows}),
            "event_recall": sum(row["detected"] for row in rows) / len(rows),
            "recourse_pass_rate": sum(row["recourse_passed"] for row in rows) / len(rows),
            "recourse_target_hit_conditional": (
                float(np.mean([row["recourse_target_hit_if_pass"] for row in rows
                               if row["recourse_target_hit_if_pass"] is not None]))
                if any(row["recourse_target_hit_if_pass"] is not None for row in rows) else None
            ),
            "recourse_success_seed_event_rows": sum(row["recourse_passed"] for row in rows),
            "recourse_target_hit_success_rows": sum(
                row["recourse_target_hit_if_pass"] is True for row in rows
            ),
            **{
                metric: {
                    "mean": float(np.mean([row[metric] for row in rows])),
                    "seed_range": [
                        float(np.mean([row[metric] for row in rows if row["seed"] == seed]))
                        for seed in sorted({row["seed"] for row in rows})
                    ],
                }
                for metric in metrics
            },
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc BATADAL pump-component target localization audit.")
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "summaries" / "batadal_target_localization.json")
    args = parser.parse_args()

    dataset = load_batadal(CONFIG_FILE)
    aligned = align_annotations_to_labeled_events(dataset.tests[0], load_target_annotations())
    all_records = []
    run_metadata = []
    seen_run_ids = set()
    seen_seeds = set()
    expected_manifest = None
    expected_models = None
    for run_dir in args.run_dirs:
        summary, records = evaluate_run(run_dir, aligned)
        manifest_identity = json.dumps(summary.get("data_manifest", {}), sort_keys=True)
        model_names = sorted(summary["models"])
        if expected_manifest is None:
            expected_manifest = manifest_identity
            expected_models = model_names
        elif manifest_identity != expected_manifest:
            raise ValueError("run directories use different dataset manifests")
        elif model_names != expected_models:
            raise ValueError("run directories contain different detector sets")
        if summary["run_id"] in seen_run_ids:
            raise ValueError(f"duplicate run ID: {summary['run_id']}")
        if int(summary["seed"]) in seen_seeds:
            raise ValueError(f"duplicate seed: {summary['seed']}")
        seen_run_ids.add(summary["run_id"])
        seen_seeds.add(int(summary["seed"]))
        run_metadata.append({"run_id": summary["run_id"], "seed": int(summary["seed"])})
        all_records.extend(records)

    payload = {
        "study": "BATADAL approximate-label pump-component target localization",
        "source_url": "https://www.batadal.net/faq.html",
        "annotation_file": str(ANNOTATION_FILE.relative_to(ROOT)),
        "label_scope": "Five approximate positive intervals; target annotations identify pump components, not exact manipulated channels or causal roots.",
        "evaluation": "Post-hoc only. Dataset annotations are never supplied to detectors or explainers. Missed events count as localization misses. Seeds are averaged only descriptively; five unique intervals do not support confirmatory inference.",
        "events": [
            {key: (value.isoformat() if isinstance(value, date) else value)
             for key, value in row.items() if key != "targets"} | {"targets": row["targets"]}
            for row in aligned
        ],
        "dataset_manifest": json.loads(expected_manifest or "{}"),
        "runs": run_metadata,
        "models": summarize(all_records),
        "records": all_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "events": len(aligned), "runs": len(run_metadata),
                      "models": payload["models"]}, indent=2))


if __name__ == "__main__":
    main()
