from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from xai_cps.data import load_hai
from xai_cps.explainers import grouped_ablation
from run_hai import model_specs


def _edited_score(detector, x: np.ndarray, dx: np.ndarray, previous: np.ndarray,
                  indices: np.ndarray, state_only: bool) -> float:
    edited = x.copy()
    edited[indices] = detector.x_ref[indices]
    edited_dx = np.zeros_like(dx) if state_only else edited - previous
    return float(detector.score(edited[None, :], edited_dx[None, :])[0])


def _random_edited_scores(detector, x: np.ndarray, dx: np.ndarray, previous: np.ndarray,
                          k: int, count: int, rng: np.random.Generator,
                          state_only: bool) -> np.ndarray:
    """Score random k-tag deletions as one detector batch."""
    selections = [rng.choice(len(x), size=k, replace=False) for _ in range(count)]
    edited = np.broadcast_to(x, (count, len(x))).copy()
    for sample, indices in enumerate(selections):
        edited[sample, indices] = detector.x_ref[indices]
    edited_dx = np.zeros_like(edited) if state_only else edited - previous
    return np.asarray(detector.score(edited, edited_dx), dtype=float)


def evaluate_run(run_dir: Path, config_path: Path, random_sets: int, seed: int) -> list[dict]:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    run_seed = int(summary["seed"])
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    dataset = load_hai(config_path, max_train_rows=config["training_rows_max"], seed=run_seed)
    event_rows = pd.read_csv(run_dir / "event_explanations.csv")
    segments = {segment.name: segment for segment in dataset.tests}
    outputs: list[dict] = []

    for model_name, detector, train_dx, _, state_only in model_specs(config, dataset):
        detector.fit(dataset.train_x, train_dx)
        rows = event_rows.loc[event_rows.model == model_name]
        for row in rows.to_dict("records"):
            segment = segments[row["file"]]
            alarm_row = int(row["explanation_row"])
            x = segment.x[alarm_row]
            previous = segment.x[alarm_row - 1] if alarm_row > 0 else x.copy()
            dx = np.zeros_like(x) if state_only else segment.dx[alarm_row]
            score = float(detector.score(x[None, :], dx[None, :])[0])
            threshold = float(row["threshold"])
            margin = score - threshold
            if not margin > 0:
                raise ValueError(f"Saved explanation point is not an alarm: {run_dir.name}/{model_name}/{row['file']}/{row['event_index']}")

            attribution, _ = grouped_ablation(
                detector, x, dx, dataset.feature_names,
                previous_x=previous, state_only=state_only)
            ranked = np.argsort(-attribution, kind="mergesort")
            rng = np.random.default_rng(
                int(seed) + run_seed * 1_000_003 + int(row["event_index"]) * 1009
                + sum(str(row["file"]).encode("utf-8")))
            for k in (1, 3, 5, 10):
                k = min(k, len(x))
                top_score = _edited_score(detector, x, dx, previous, ranked[:k], state_only)
                bottom_score = _edited_score(detector, x, dx, previous, ranked[-k:], state_only)
                random_scores = _random_edited_scores(
                    detector, x, dx, previous, k, random_sets, rng, state_only)
                outputs.append({
                    "run_id": run_dir.name,
                    "seed": run_seed,
                    "model": model_name,
                    "file": row["file"],
                    "event_index": int(row["event_index"]),
                    "alarm_row": alarm_row,
                    "k": k,
                    "score": score,
                    "threshold": threshold,
                    "alarm_margin": margin,
                    "topk_margin_removed": float((score - top_score) / margin),
                    "bottomk_margin_removed": float((score - bottom_score) / margin),
                    "random_margin_removed_mean": float(np.mean((score - random_scores) / margin)),
                    "topk_threshold_crossed": bool(top_score <= threshold),
                    "bottomk_threshold_crossed": bool(bottom_score <= threshold),
                    "random_threshold_crossed_rate": float(np.mean(random_scores <= threshold)),
                    "state_only": bool(state_only),
                    "temporal_identity_preserved": bool(
                        state_only or np.allclose(dx, x - previous, rtol=0.0, atol=1e-10)
                    ),
                })
    return outputs


def summarize(rows: list[dict], draws: int, seed: int) -> dict:
    frame = pd.DataFrame(rows)
    rng = np.random.default_rng(seed)
    per_event = frame.groupby(["model", "k", "file", "event_index"], as_index=False).agg(
        topk_margin_removed=("topk_margin_removed", "mean"),
        bottomk_margin_removed=("bottomk_margin_removed", "mean"),
        random_margin_removed_mean=("random_margin_removed_mean", "mean"),
        topk_threshold_crossed=("topk_threshold_crossed", "mean"),
        random_threshold_crossed_rate=("random_threshold_crossed_rate", "mean"),
    )
    results: dict[str, dict] = {}
    tests: list[tuple[str, str, float]] = []
    for (model, k), group in per_event.groupby(["model", "k"], sort=True):
        delta = (group["topk_margin_removed"] - group["random_margin_removed_mean"]).to_numpy(float)
        boot = np.asarray([
            np.mean(rng.choice(delta, size=len(delta), replace=True)) for _ in range(draws)
        ])
        observed = float(np.mean(delta))
        signs = np.asarray([
            np.mean(delta * rng.choice(np.array([-1.0, 1.0]), size=len(delta))) for _ in range(draws)
        ])
        p_value = float((np.sum(np.abs(signs) >= abs(observed)) + 1) / (draws + 1))
        test_name = f"{model}|k={k}"
        tests.append((model, test_name, p_value))
        results.setdefault(model, {})[str(k)] = {
            "n_event_ids_after_seed_averaging": int(len(group)),
            "topk_alarm_margin_removed_mean": float(group["topk_margin_removed"].mean()),
            "bottomk_alarm_margin_removed_mean": float(group["bottomk_margin_removed"].mean()),
            "random_alarm_margin_removed_mean": float(group["random_margin_removed_mean"].mean()),
            "topk_threshold_crossing_rate": float(group["topk_threshold_crossed"].mean()),
            "random_threshold_crossing_rate": float(group["random_threshold_crossed_rate"].mean()),
            "topk_minus_random_margin_removed": observed,
            "event_bootstrap_ci95": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
            "paired_sign_flip_p_two_sided": p_value,
            "interpretation": "detector-score deletion sufficiency relative to the alarm margin; not explanation correctness or causal validity",
        }
    ordered = sorted(tests, key=lambda x: x[2])
    running = 0.0
    for rank, (model, test_name, p_value) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * p_value))
        k = test_name.rsplit("k=", 1)[1]
        results[model][k]["holm_adjusted_p_two_sided"] = running
    return {
        "dataset": "HAI 21.03",
        "design": "At each saved first-alarm point, replace top-ranked, bottom-ranked, or random k sensor states with training-normal references; temporal deltas are recomputed against the observed preceding state and state-only deltas remain zero.",
        "metric": "fraction of the detector's score margin above threshold removed by a joint reference deletion; 1.0 reaches or crosses the threshold",
        "seeds_averaged_within_event_before_bootstrap": True,
        "random_sets_per_event_and_k": None,
        "models": results,
        "event_seed_rows": rows,
        "limitations": "The random-set comparator is Monte Carlo sampled, and its Monte Carlo uncertainty is not nested inside the event bootstrap. This evaluates the detector's local response to reference substitution; it is neither causal ground truth nor physical intervention feasibility.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate explanation deletion sufficiency at HAI first-alarm points.")
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "hai_21_03.json")
    parser.add_argument("--random-sets", type=int, default=50)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.random_sets < 1 or args.bootstrap_draws < 1:
        parser.error("random sets and bootstrap draws must be positive")
    rows = []
    for run in args.runs:
        rows.extend(evaluate_run(run, args.config, args.random_sets, args.seed))
    result = summarize(rows, args.bootstrap_draws, args.seed)
    result["random_sets_per_event_and_k"] = int(args.random_sets)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "models": result["models"]}, indent=2))


if __name__ == "__main__":
    main()
