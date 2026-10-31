from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def finite_mean(values):
    x=np.asarray([v for v in values if v is not None and np.isfinite(v)],dtype=float)
    return float(x.mean()) if len(x) else None


def finite_max(values):
    x=np.asarray([v for v in values if v is not None and np.isfinite(v)],dtype=float)
    return float(x.max()) if len(x) else None


def summarize(run_dirs: list[Path]) -> dict:
    runs=[]
    for run_dir in run_dirs:
        summary=json.loads((run_dir/"summary.json").read_text(encoding="utf-8"))
        events=pd.read_csv(run_dir/"event_explanations.csv")
        models={}
        for model,metrics in summary["models"].items():
            rows=events[events.model==model]
            success=rows[rows.capability_support_success.astype(bool)]
            cf=metrics["counterfactual"]
            models[model]={
                "detector_f1":metrics["overall_point_metrics"]["f1"],
                "event_recall":metrics["overall_event_metrics"]["event_recall"],
                "false_alarm_events_per_normal_hour":metrics["overall_event_metrics"]["false_alarm_events_per_normal_hour"],
                "alarm_events_explained":int(len(rows)),
                "capability_support_success_rate":cf["capability_support_score_flip_rate"],
                "dynamics_constrained_success_rate":cf.get("dynamics_constrained_score_flip_support_rate"),
                "dynamics_constrained_stage_hit_at_1_on_success":cf.get("dynamics_constrained_stage_hit_at_1_given_success"),
                "dynamics_constrained_median_edit_count_on_success":cf.get("dynamics_constrained_median_edit_count_on_success"),
                "dynamics_constrained_transition_threshold":cf.get("dynamics_validation_q95_threshold"),
                "greedy_scoreflip_and_support_rate":finite_mean(
                    rows.counterfactual_all_success.astype(float)*rows.counterfactual_within_validation_reference_q95.astype(float)),
                "capability_group_budget_success_rate":cf["capability_success_rate_by_group_budget"],
                "capability_tag_budget_success_rate":cf["capability_success_rate_by_tag_budget"],
                "median_edited_tags_on_success":float(success.capability_edit_count.median()) if len(success) else None,
                "median_norm_cost_proposed_on_success":float(success.capability_normalized_l1.median()) if len(success) else None,
                "median_norm_cost_greedy_on_same_events":float(success.counterfactual_all_normalized_l1.median()) if len(success) else None,
                "stage_hit_at_1_on_success":finite_mean(success.capability_stage_hit_at_1_after_generation),
                "transition_consistency_max_abs_error_on_success":(
                    None if model=="PCA-STATE" else finite_max(success.capability_transition_consistency_max_abs_error)
                ),
                "false_alarm_explanation_sample":metrics.get("false_alarm_explanation_sample", {}),
                "dynamics_constrained_search_p95_ms":metrics["latency"].get("dynamics_constrained_search_p95_ms"),
                "explanation_p95_ms":metrics["latency"]["per_event_explanation_p95_ms"],
            }
        runs.append({"run_id":summary["run_id"],"seed":summary["seed"],"models":models})

    model_names=sorted({m for run in runs for m in run["models"]})
    aggregate={}
    scalar_keys=["detector_f1","event_recall","false_alarm_events_per_normal_hour",
                 "capability_support_success_rate","dynamics_constrained_success_rate",
                 "dynamics_constrained_stage_hit_at_1_on_success",
                 "dynamics_constrained_median_edit_count_on_success","dynamics_constrained_transition_threshold",
                 "dynamics_constrained_search_p95_ms",
                 "greedy_scoreflip_and_support_rate",
                 "median_edited_tags_on_success","median_norm_cost_proposed_on_success",
                 "median_norm_cost_greedy_on_same_events","stage_hit_at_1_on_success",
                 "transition_consistency_max_abs_error_on_success","explanation_p95_ms"]
    for model in model_names:
        aggregate[model]={}
        for key in scalar_keys:
            values=[run["models"][model][key] for run in runs if model in run["models"]]
            vals=[v for v in values if v is not None and np.isfinite(v)]
            aggregate[model][key]={"mean":float(np.mean(vals)) if vals else None,
                                   "range":[float(np.min(vals)),float(np.max(vals))] if vals else None}
        for key in ("capability_group_budget_success_rate","capability_tag_budget_success_rate"):
            budget_keys=sorted({b for run in runs if model in run["models"] for b in run["models"][model][key]},key=int)
            aggregate[model][key]={b:{"mean":finite_mean([run["models"][model][key].get(b) for run in runs if model in run["models"]]),
                                      "range":[min(v),max(v)] if (v:=[run["models"][model][key][b] for run in runs if model in run["models"] and b in run["models"][model][key]]) else None}
                                     for b in budget_keys}
        fa_runs=[run["models"][model]["false_alarm_explanation_sample"] for run in runs
                 if model in run["models"] and run["models"][model]["false_alarm_explanation_sample"]]
        if fa_runs:
            fa_keys=["total_pure_false_alarm_episodes","sampled_episodes","sample_fraction",
                     "sampled_greedy_score_flip_rate","sampled_capability_support_success_rate",
                     "sampled_dynamics_constrained_success_rate",
                     "median_edited_tags_on_success","explanation_latency_p50_ms","explanation_latency_p95_ms",
                     "dynamics_explanation_latency_p95_ms"]
            aggregate[model]["false_alarm_explanation_sample"]={}
            for key in fa_keys:
                vals=[item[key] for item in fa_runs if item.get(key) is not None and np.isfinite(item[key])]
                aggregate[model]["false_alarm_explanation_sample"][key]={
                    "mean":float(np.mean(vals)) if vals else None,
                    "range":[float(np.min(vals)),float(np.max(vals))] if vals else None,
                }
    return {"runs":[{"run_id":r["run_id"],"seed":r["seed"]} for r in runs],"models":aggregate,"per_run":runs,
            "interpretation":"Descriptive aggregation across supplied runs only; ranges are not confidence intervals."}


def main():
    parser=argparse.ArgumentParser(description="Aggregate saved HAI explanation run artifacts.")
    parser.add_argument("runs",nargs="+",type=Path,help="run directories containing summary.json and event_explanations.csv")
    parser.add_argument("--output",type=Path,help="optional JSON output path")
    args=parser.parse_args()
    result=summarize(args.runs)
    encoded=json.dumps(result,indent=2,allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(encoded+"\n",encoding="utf-8")
    print(encoded)


if __name__=="__main__":
    main()
