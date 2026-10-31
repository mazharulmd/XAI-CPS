from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from xai_cps.threshold_sensitivity import aggregate_runs


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine independently executed per-seed threshold audits")
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "summaries" / "hai21_threshold_sensitivity_seed42_46.json")
    args = parser.parse_args()
    inputs = [json.loads(path.read_text(encoding="utf-8")) for path in args.summaries]
    runs = [run for report in inputs for run in report["runs"]]
    seeds = [int(run["seed"]) for run in runs]
    if len(seeds) != len(set(seeds)):
        raise ValueError("input summaries contain duplicate seeds")
    quantile_sets = {tuple(report["quantiles"]) for report in inputs}
    if len(quantile_sets) != 1:
        raise ValueError("all input summaries must use the same quantile grid")
    report = {
        "study": "HAI 21.03 validation-quantile threshold sensitivity",
        "source_run_ids": [run["run_id"] for run in runs],
        "seeds": seeds,
        "quantiles": list(next(iter(quantile_sets))),
        "threshold_calibration": "each run's normal-only train sample and train3 validation scores; test scores and labels are never used to select thresholds",
        "test_score_source": "saved raw scores from each specified frozen seed run; detector models are refit only to reconstruct validation quantiles",
        "explanations_recomputed": False,
        "aggregation_note": "means and ranges summarize repeated model fits on the same five held-out files and attack events; they are not independent test replications or confidence intervals",
        "runs": runs,
        "mean_and_seed_range": aggregate_runs(runs),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "seeds": seeds, "summary": report["mean_and_seed_range"]}, indent=2))


if __name__ == "__main__":
    main()
