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


def topk_jaccard(a: np.ndarray, b: np.ndarray, k: int = 5) -> float:
    left = set(np.argsort(-a)[:k].tolist())
    right = set(np.argsort(-b)[:k].tolist())
    return len(left & right) / max(1, len(left | right))


def evaluate_run(run_dir: Path, config_path: Path, sigmas: tuple[float, ...], perturbations: int) -> list[dict]:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    seed = int(summary["seed"])
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    dataset = load_hai(config_path, max_train_rows=config["training_rows_max"], seed=seed)
    event_rows = pd.read_csv(run_dir / "event_explanations.csv")
    event_rows = event_rows.loc[event_rows["top5_stability_mean"].notna()].copy()
    segments = {segment.name: segment for segment in dataset.tests}
    outputs: list[dict] = []

    for model_name, detector, train_dx, _, state_only in model_specs(config, dataset):
        detector.fit(dataset.train_x, train_dx)
        rows = event_rows.loc[event_rows.model == model_name]
        for row in rows.to_dict("records"):
            segment = segments[row["file"]]
            current_row = int(row["explanation_row"])
            x = segment.x[current_row]
            previous = segment.x[current_row - 1] if current_row > 0 else x.copy()
            dx = np.zeros_like(x) if state_only else segment.dx[current_row]
            if not state_only:
                np.testing.assert_allclose(dx, x - previous, rtol=0.0, atol=1e-10)
            # Keep each explainer's baseline fixed while varying input perturbation semantics.
            # This makes input design and ablation design a crossed comparison.
            base_legacy, _ = grouped_ablation(
                detector, x, dx, dataset.feature_names, preserve_temporal=False)
            base_coherent, _ = grouped_ablation(
                detector, x, dx, dataset.feature_names,
                previous_x=previous, state_only=state_only)
            rng = np.random.default_rng(seed + int(row["peak_row"]) + int(row["event_index"]))
            legacy_invalid_scores: list[float] = []
            legacy_valid_scores: list[float] = []
            coherent_invalid_scores: list[float] = []
            coherent_by_sigma = {sigma: [] for sigma in sigmas}

            for _ in range(perturbations):
                eps_x = rng.normal(0.0, 1.0, size=x.shape)
                # Reproduce the previous implementation exactly at its 2% setting.
                eps_dx = rng.normal(0.0, 1.0, size=dx.shape)
                x_legacy = x + 0.02 * eps_x * detector.x_scale
                dx_legacy = dx + 0.02 * eps_dx * detector.dx_scale
                legacy_invalid, _ = grouped_ablation(
                    detector, x_legacy, dx_legacy, dataset.feature_names,
                    preserve_temporal=False)
                legacy_invalid_scores.append(topk_jaccard(base_legacy, legacy_invalid))
                if not state_only:
                    # Same legacy ablation explainer, but a representation-valid perturbed input.
                    x_valid = x + 0.02 * eps_x * detector.x_scale
                    dx_valid = x_valid - previous
                    legacy_valid, _ = grouped_ablation(
                        detector, x_valid, dx_valid, dataset.feature_names,
                        preserve_temporal=False)
                    legacy_valid_scores.append(topk_jaccard(base_legacy, legacy_valid))
                    # Same identity-preserving ablation explainer, applied to the invalid input.
                    coherent_invalid, _ = grouped_ablation(
                        detector, x_legacy, dx_legacy, dataset.feature_names,
                        previous_x=previous, state_only=False)
                    coherent_invalid_scores.append(topk_jaccard(base_coherent, coherent_invalid))

                for sigma in sigmas:
                    x_perturbed = x + sigma * eps_x * detector.x_scale
                    dx_perturbed = np.zeros_like(dx) if state_only else x_perturbed - previous
                    if state_only:
                        assert np.all(dx_perturbed == 0.0)
                    else:
                        np.testing.assert_allclose(dx_perturbed, x_perturbed - previous,
                                                   rtol=0.0, atol=1e-10)
                    coherent, _ = grouped_ablation(
                        detector, x_perturbed, dx_perturbed, dataset.feature_names,
                        previous_x=previous, state_only=state_only)
                    coherent_by_sigma[sigma].append(topk_jaccard(base_coherent, coherent))

            result = {
                "run_id": run_dir.name,
                "seed": seed,
                "model": model_name,
                "file": row["file"],
                "event_index": int(row["event_index"]),
                "explanation_row": current_row,
                "legacy_explainer_invalid_input_2pct_jaccard": float(np.mean(legacy_invalid_scores)),
                "legacy_explainer_valid_input_2pct_jaccard": float(np.mean(legacy_valid_scores)) if legacy_valid_scores else None,
                "identity_preserving_explainer_invalid_input_2pct_jaccard": float(np.mean(coherent_invalid_scores)) if coherent_invalid_scores else None,
                "perturbations": int(perturbations),
                "state_only": bool(state_only),
            }
            for sigma, values in coherent_by_sigma.items():
                result[f"identity_preserving_{sigma:g}_jaccard"] = float(np.mean(values))
            outputs.append(result)
    return outputs


def aggregate(rows: list[dict], sigmas: tuple[float, ...], draws: int, seed: int) -> dict:
    frame = pd.DataFrame(rows)
    metrics = [
        "legacy_explainer_invalid_input_2pct_jaccard",
        "legacy_explainer_valid_input_2pct_jaccard",
        "identity_preserving_explainer_invalid_input_2pct_jaccard",
    ] + [f"identity_preserving_{sigma:g}_jaccard" for sigma in sigmas]
    per_run = []
    for (run_id, run_seed, model), group in frame.groupby(["run_id", "seed", "model"], sort=True):
        values = {metric: (float(group[metric].mean()) if group[metric].notna().any() else None) for metric in metrics}
        per_run.append({"run_id": run_id, "seed": int(run_seed), "model": model,
                        "n_alarm_events": int(len(group)), "mean_jaccard": values})

    aggregate = {}
    paired_comparisons = {}
    rng = np.random.default_rng(seed)
    for model, model_rows in frame.groupby("model", sort=True):
        aggregate[model] = {"n_event_seed_records": int(len(model_rows)),
                            "n_unique_events": int(model_rows[["file", "event_index"]].drop_duplicates().shape[0]),
                            "metrics": {}}
        for metric in metrics:
            run_values = [item["mean_jaccard"][metric] for item in per_run
                          if item["model"] == model and item["mean_jaccard"][metric] is not None]
            by_event = model_rows.groupby(["file", "event_index"], sort=True)[metric].mean().to_numpy(dtype=float)
            by_event = by_event[np.isfinite(by_event)]
            if len(by_event):
                boot = np.asarray([np.mean(rng.choice(by_event, size=len(by_event), replace=True))
                                   for _ in range(draws)])
                ci = [float(v) for v in np.quantile(boot, [0.025, 0.975])]
                event_mean = float(np.mean(by_event))
            else:
                ci, event_mean = None, None
            aggregate[model]["metrics"][metric] = {
                "mean_across_event_seed_records": (float(model_rows[metric].mean()) if model_rows[metric].notna().any() else None),
                "mean_after_averaging_seeds_within_event": event_mean,
                "event_bootstrap_ci95_after_seed_averaging": ci,
                "seed_mean_range": [float(min(run_values)), float(max(run_values))] if run_values else None,
            }
        event_means = model_rows.groupby(["file", "event_index"], sort=True)[metrics].mean()
        paired_comparisons[model] = {}
        comparisons = []
        if 0.02 in sigmas and not model_rows["state_only"].all():
            legacy_effect = (event_means["legacy_explainer_invalid_input_2pct_jaccard"]
                             - event_means["legacy_explainer_valid_input_2pct_jaccard"])
            coherent_effect = (event_means["identity_preserving_explainer_invalid_input_2pct_jaccard"]
                               - event_means["identity_preserving_0.02_jaccard"])
            comparisons.append(("legacy_explainer_invalid_minus_valid_input_2pct", legacy_effect.to_numpy()))
            comparisons.append(("identity_preserving_explainer_invalid_minus_valid_input_2pct", coherent_effect.to_numpy()))
            comparisons.append(("difference_in_input_design_effect_by_ablation_semantics", (legacy_effect - coherent_effect).to_numpy()))
        if 0.02 in sigmas and 0.05 in sigmas:
            differences = (event_means["identity_preserving_0.05_jaccard"]
                           - event_means["identity_preserving_0.02_jaccard"]).to_numpy()
            comparisons.append(("identity_preserving_0.05_minus_identity_preserving_0.02", differences))
        for comparison_name, differences in comparisons:
            if len(differences):
                event_boot = np.asarray([
                    np.mean(rng.choice(differences, size=len(differences), replace=True))
                    for _ in range(draws)
                ])
                sign_flips = np.asarray([
                    np.mean(differences * rng.choice(np.array([-1.0, 1.0]), size=len(differences)))
                    for _ in range(draws)
                ])
                observed = float(np.mean(differences))
                p_value = float((np.sum(np.abs(sign_flips) >= abs(observed)) + 1) / (draws + 1))
                ci = [float(value) for value in np.quantile(event_boot, [0.025, 0.975])]
            else:
                observed, p_value, ci = None, None, None
            paired_comparisons[model][comparison_name] = {
                "mean_paired_event_difference": observed,
                "event_bootstrap_ci95": ci,
                "paired_sign_flip_p_two_sided": p_value,
                "n_matched_events": int(len(differences)),
                "bootstrap_draws": int(draws),
                "estimand": "paired top-five grouped-ablation Jaccard difference with the explainer baseline fixed within each comparison; seeds averaged within repeated HAI event ID before resampling",
            }
    flat_tests = [(model, name, result["paired_sign_flip_p_two_sided"])
                  for model, comparisons in paired_comparisons.items()
                  for name, result in comparisons.items()
                  if result["paired_sign_flip_p_two_sided"] is not None]
    ordered = sorted(flat_tests, key=lambda item: item[2])
    m = len(ordered)
    running = 0.0
    for rank, (model, name, p_value) in enumerate(ordered):
        adjusted = min(1.0, (m - rank) * p_value)
        running = max(running, adjusted)
        paired_comparisons[model][name]["holm_adjusted_p_two_sided"] = running
        paired_comparisons[model][name]["holm_family_size"] = m
    return {
        "dataset": "HAI 21.03",
        "runs": sorted({row["run_id"] for row in rows}),
        "perturbation_design": {
            "factorial_design": "crosses input perturbation semantics (independent state/difference versus identity-preserving state jitter) with grouped-ablation semantics (legacy independent channel edits versus identity-preserving channel edits); paired input-design contrasts hold the explainer baseline fixed",
            "identity_preserving": "Gaussian current-state jitter and temporal ablation attributions recompute first differences against the observed preceding state; PCA-STATE delta remains zero and is excluded from temporal input-design contrasts",
            "sigmas": list(sigmas), "perturbations_per_event_and_design": int(rows[0].get("perturbations", 0) if rows else 0),
            "events": "the same detected alarm points with a non-null prior top5_stability_mean field, up to the first ten attack intervals per test file",
        },
        "models": aggregate,
        "paired_comparisons": paired_comparisons,
        "per_run": per_run,
        "event_rows": rows,
        "interpretation": "Stability is top-five grouped-ablation rank overlap under synthetic small measurement noise. It is not robustness to adaptive attacks, distribution shift, explanation correctness, or causal validity. Seed ranges are descriptive; event-bootstrap intervals average seeds within repeated scenario IDs first.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit HAI explanation stability under representation-consistent perturbations.")
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "hai_21_03.json")
    parser.add_argument("--sigmas", nargs="+", type=float, default=[0.02, 0.05])
    parser.add_argument("--perturbations", type=int, default=5)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.perturbations < 1 or any(sigma <= 0.0 for sigma in args.sigmas):
        parser.error("perturbations and all sigma values must be positive")
    rows = []
    for run in args.runs:
        rows.extend(evaluate_run(run, args.config, tuple(args.sigmas), args.perturbations))
    result = aggregate(rows, tuple(args.sigmas), args.bootstrap_draws, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "models": result["models"]}, indent=2))


if __name__ == "__main__":
    main()
