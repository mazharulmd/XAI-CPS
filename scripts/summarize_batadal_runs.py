from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser(description="Aggregate partial-label BATADAL run summaries.")
    parser.add_argument("runs",nargs="+",type=Path)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    runs=[json.loads((p/"summary.json").read_text(encoding="utf-8")) for p in args.runs]
    names=sorted({name for run in runs for name in run["models"]})
    aggregate={}
    metric_fields=("known_attack_positive_point_recall","approx_labeled_event_recall",
                   "approx_label_mean_detection_delay_seconds","capability_support_scoreflip_rate",
                   "validation_point_fpr","explanation_p95_ms")
    for name in names:
        aggregate[name]={}
        for field in metric_fields:
            values=[run["models"][name][field] for run in runs if name in run["models"]]
            values=[float(v) for v in values if v is not None and np.isfinite(v)]
            aggregate[name][field]={"mean":float(np.mean(values)) if values else None,
                                    "range":[float(min(values)),float(max(values))] if values else None}
        aggregate[name]["labeled_event_counts"]=[run["models"][name]["approx_labeled_attack_events"] for run in runs]
        aggregate[name]["test_precision_f1_fpr"]="not estimable: all non-positive ATT_FLAG rows are unknown"
    output={"runs":[{"run_id":run["run_id"],"seed":run["seed"]} for run in runs],
            "label_scope":"Approximate positive annotations only; unknown rows are not treated as negatives.",
            "models":aggregate,
            "interpretation":"Descriptive aggregation only; two runs and five labeled intervals do not support inferential claims."}
    encoded=json.dumps(output,indent=2,allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(encoded+"\n",encoding="utf-8")
    print(encoded)


if __name__=="__main__": main()
