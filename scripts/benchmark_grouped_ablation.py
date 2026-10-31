from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from xai_cps.data import load_hai
from xai_cps.explainers import grouped_ablation
from run_hai import model_specs


def loop_ablation(detector, x, dx, feature_names, previous_x, state_only):
    base = float(detector.score(x[None, :], dx[None, :])[0])
    result = np.zeros(len(feature_names), dtype=float)
    for j in range(len(feature_names)):
        edited = x.copy()
        edited[j] = detector.x_ref[j]
        edited_dx = np.zeros_like(dx) if state_only else edited - previous_x
        result[j] = base - float(detector.score(edited[None, :], edited_dx[None, :])[0])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired latency and numerical-equivalence check for grouped ablation implementations.")
    parser.add_argument("run", type=Path, help="run directory with saved first-alarm rows")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "hai_21_03.json")
    parser.add_argument("--events-per-model", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.run / "summary.json").read_text(encoding="utf-8"))
    seed = int(summary["seed"])
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    data = load_hai(args.config, max_train_rows=config["training_rows_max"], seed=seed)
    segments = {segment.name: segment for segment in data.tests}
    events = pd.read_csv(args.run / "event_explanations.csv")
    results = []
    for model_name, detector, train_dx, _, state_only in model_specs(config, data):
        detector.fit(data.train_x, train_dx)
        rows = events.loc[events.model == model_name].head(args.events_per_model)
        for event in rows.to_dict("records"):
            seg = segments[event["file"]]
            row = int(event["explanation_row"])
            x = seg.x[row]
            dx = np.zeros_like(x) if state_only else seg.dx[row]
            previous = seg.x[row - 1] if row else x.copy()
            expected = loop_ablation(detector, x, dx, data.feature_names, previous, state_only)
            measured_loop = []
            measured_batch = []
            for repeat in range(args.repeats + 1):
                started = time.perf_counter()
                batch_values, _ = grouped_ablation(
                    detector, x, dx, data.feature_names,
                    previous_x=previous, state_only=state_only)
                batch_time = (time.perf_counter() - started) * 1000.0
                started = time.perf_counter()
                loop_values = loop_ablation(detector, x, dx, data.feature_names, previous, state_only)
                loop_time = (time.perf_counter() - started) * 1000.0
                if repeat:
                    measured_batch.append(batch_time)
                    measured_loop.append(loop_time)
                max_error = float(np.max(np.abs(batch_values - expected)))
                if max_error > 1e-9:
                    raise AssertionError(f"Vectorized attribution mismatch: {model_name} {event['file']} event {event['event_index']} max={max_error}")
            results.append({
                "model": model_name,
                "file": event["file"],
                "event_index": int(event["event_index"]),
                "vectorized_p50_ms": float(np.median(measured_batch)),
                "loop_p50_ms": float(np.median(measured_loop)),
                "vectorized_median_speed_ratio_vs_loop": float(np.median(measured_loop) / max(np.median(measured_batch), 1e-12)),
                "max_attribution_abs_difference": max_error,
                "state_only": bool(state_only),
            })
    payload = {
        "run_id": args.run.name,
        "paired_design": "same event, fitted seed-specific detector, and reference values; warmup excluded; vectorized and identity-preserving loop implementations alternated per repeat",
        "events_per_model_max": int(args.events_per_model),
        "repeats_per_event_after_warmup": int(args.repeats),
        "event_results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for model in sorted({r["model"] for r in results}):
        subset = [r for r in results if r["model"] == model]
        print(model, {
            "n_events": len(subset),
            "loop_median_ms": float(np.median([r["loop_p50_ms"] for r in subset])),
            "vectorized_median_ms": float(np.median([r["vectorized_p50_ms"] for r in subset])),
            "speed_ratio": float(np.median([r["vectorized_median_speed_ratio_vs_loop"] for r in subset])),
            "max_abs_difference": max(r["max_attribution_abs_difference"] for r in subset),
        })


if __name__ == "__main__":
    main()
