from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_hai_explanation_stability import aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate factorial representation-stability event rows across HAI seeds")
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.summaries]
    event_rows = [row for report in reports for row in report["event_rows"]]
    run_ids = sorted({row["run_id"] for row in event_rows})
    seeds = sorted({int(row["seed"]) for row in event_rows})
    if len(run_ids) != len(seeds):
        raise ValueError("expected one unique run per seed in the aggregate")
    sigmas = tuple(float(value) for value in reports[0]["perturbation_design"]["sigmas"])
    if any(tuple(float(value) for value in report["perturbation_design"]["sigmas"]) != sigmas for report in reports):
        raise ValueError("all per-seed reports must use the same jitter scales")
    result = aggregate(event_rows, sigmas, args.bootstrap_draws, args.seed)
    result["seed_list"] = seeds
    result["aggregation_note"] = "Repeated seeds reuse the same event IDs; event bootstrap averages seeds within event before resampling and five-file cluster uncertainty is not resolved."
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "seeds": seeds,
                      "paired_comparisons": result["paired_comparisons"]}, indent=2))


if __name__ == "__main__":
    main()
