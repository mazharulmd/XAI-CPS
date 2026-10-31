from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from xai_cps.data import contiguous_events, load_hai
from xai_cps.detectors import valid_references
from xai_cps.explainers import grouped_ablation
from xai_cps.isolation_forest import HAIIsolationForest
from xai_cps.metrics import binary_metrics, event_metrics
from xai_cps.target_eval import random_hit_probability


def first_alarm_targets(
    detector: HAIIsolationForest,
    segment,
    scores: np.ndarray,
    threshold: float,
    feature_names: list[str],
    annotations: pd.DataFrame,
) -> list[dict[str, object]]:
    results = []
    detected_rows = np.flatnonzero(scores > threshold)
    for row in annotations.to_dict("records"):
        start = int(row["start_row"])
        end = int(row["end_row_exclusive"])
        alarm_rows = detected_rows[(detected_rows >= start) & (detected_rows < end)]
        alarm = int(alarm_rows[0]) if len(alarm_rows) else None
        targets = sorted(set(str(row["target_points_present_in_csv"]).split("|")) - {""})
        if alarm is None:
            ranked = []
        else:
            previous = segment.x[alarm - 1] if alarm else segment.x[alarm]
            attribution, _ = grouped_ablation(
                detector,
                segment.x[alarm],
                segment.dx[alarm],
                feature_names,
                previous_x=previous,
            )
            ranked = [feature_names[j] for j in np.argsort(-attribution, kind="stable")]
        record: dict[str, object] = {
            "attack_id": str(row["attack_id"]),
            "dataset_file": str(row["dataset_file"]),
            "event_index": int(row["event_index"]),
            "detected": alarm is not None,
            "first_alarm_row": alarm,
            "target_tags": targets,
            "ranked_tags": ranked[:10],
            "ranked_feature_count": len(feature_names),
        }
        for k in (1, 3, 5):
            record[f"hit_at_{k}"] = bool(set(targets) & set(ranked[:k]))
            record[f"random_hit_probability_at_{k}"] = (
                random_hit_probability(len(targets), len(feature_names), k)
                if alarm is not None else 0.0
            )
        results.append(record)
    return results


def score_in_batches(detector: HAIIsolationForest, x: np.ndarray, dx: np.ndarray, batch_size: int = 50000) -> np.ndarray:
    parts = [detector.score(x[start:start + batch_size], dx[start:start + batch_size])
             for start in range(0, len(x), batch_size)]
    return np.concatenate(parts) if parts else np.empty(0, dtype=float)


def clustered_comparison(
    if_events: list[dict[str, object]],
    reference_events: list[dict[str, object]],
    seed: int,
    draws: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    ref = {str(row["attack_id"]): row for row in reference_events}
    own = {str(row["attack_id"]): row for row in if_events}
    ids = sorted(set(ref) & set(own))
    output = {}
    raw_p = {}
    for k in (1, 3, 5):
        metric = f"hit_at_{k}"
        diff = np.asarray([float(own[i][metric]) - float(ref[i][metric]) for i in ids])
        observed = float(diff.mean())
        boot = np.mean(diff[rng.integers(0, len(diff), size=(draws, len(diff)))], axis=1)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(draws, len(diff)))
        null = np.mean(diff[None, :] * signs, axis=1)
        p = float((1 + np.count_nonzero(np.abs(null) >= abs(observed))) / (draws + 1))
        raw_p[str(k)] = p
        output[str(k)] = {
            "estimand": "IF grouped-ablation minus PCA-TEMPORAL grouped-ablation target hit rate; 29 exact-target events",
            "mean_paired_difference": observed,
            "event_cluster_bootstrap_95pct_ci": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
            "monte_carlo_sign_flip_p_two_sided": p,
            "n_events": len(ids),
            "draws": draws,
        }
    # Holm step-down adjustment over the three predeclared Hit@K comparisons.
    ordered = sorted(raw_p, key=raw_p.get)
    adjusted = {}
    running = 0.0
    for rank, key in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * raw_p[key]))
        adjusted[key] = running
    for key, value in output.items():
        value["holm_adjusted_p"] = adjusted[key]
    return output


def comparison_from_events(event_records: list[dict[str, object]], prior_path: Path, draws: int) -> dict[str, object]:
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    prior_rows = [row for run in prior["runs"] for row in run["event_outcomes"]["PCA-TEMPORAL"]]
    prior_by_attack = {}
    for attack_id in sorted({row["attack_id"] for row in prior_rows}):
        these = [row for row in prior_rows if row["attack_id"] == attack_id]
        prior_by_attack[attack_id] = {
            f"hit_at_{k}": float(np.mean([row[f"grouped_ablation_hit_at_{k}"] for row in these]))
            for k in (1, 3, 5)
        }
    mean_if_by_attack = {}
    for attack_id in sorted({row["attack_id"] for row in event_records}):
        these = [row for row in event_records if row["attack_id"] == attack_id]
        mean_if_by_attack[attack_id] = {
            f"hit_at_{k}": float(np.mean([row[f"hit_at_{k}"] for row in these]))
            for k in (1, 3, 5)
        }
    if_events = [dict(attack_id=key, **values) for key, values in mean_if_by_attack.items()]
    pca_events = [dict(attack_id=key, **values) for key, values in prior_by_attack.items()]
    return clustered_comparison(if_events, pca_events, seed=271828, draws=draws)


def random_ranking_benchmark(event_records: list[dict[str, object]], draws: int, seed: int = 161803) -> dict[str, object]:
    """Conditional random top-k rank null, preserving detection/miss status per seed-event."""
    rng = np.random.default_rng(seed)
    attack_ids = sorted({str(row["attack_id"]) for row in event_records})
    by_event = {attack_id: [row for row in event_records if str(row["attack_id"]) == attack_id]
                for attack_id in attack_ids}
    result = {}
    for k in (1, 3, 5):
        observed = np.asarray([
            np.mean([bool(row[f"hit_at_{k}"]) for row in by_event[attack_id]])
            for attack_id in attack_ids
        ], dtype=float)
        probabilities = np.asarray([
            [float(row[f"random_hit_probability_at_{k}"]) for row in by_event[attack_id]]
            for attack_id in attack_ids
        ], dtype=float)
        expected = probabilities.mean(axis=1)
        difference = observed - expected
        bootstrap = difference[rng.integers(0, len(difference), size=(draws, len(difference)))].mean(axis=1)
        random_hits = rng.random((draws, len(attack_ids), probabilities.shape[1])) < probabilities[None, :, :]
        null = random_hits.mean(axis=2).mean(axis=1)
        observed_rate = float(observed.mean())
        p_value = float((1 + np.count_nonzero(null >= observed_rate)) / (draws + 1))
        result[str(k)] = {
            "estimand": "unconditional Hit@K over 29 unique attack events; each seed-event is randomized only if the detector fired",
            "observed_mean_hit_rate": observed_rate,
            "event_matched_random_expected_hit_rate": float(expected.mean()),
            "observed_minus_random_expected": float(difference.mean()),
            "event_cluster_bootstrap_95pct_ci_for_difference": [float(x) for x in np.quantile(bootstrap, [0.025, 0.975])],
            "conditional_random_ranking_p_one_sided": p_value,
            "n_events": len(attack_ids),
            "n_repeated_fits_per_event": probabilities.shape[1],
            "draws": draws,
        }
    # Three predeclared Hit@K tests, Holm step-down correction.
    ordered = sorted(result, key=lambda key: result[key]["conditional_random_ranking_p_one_sided"])
    running = 0.0
    for rank, key in enumerate(ordered):
        raw = result[key]["conditional_random_ranking_p_one_sided"]
        running = max(running, min(1.0, (len(ordered) - rank) * raw))
        result[key]["holm_adjusted_p"] = running
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="HAI 21.03 normal-only Isolation Forest target-localization reference")
    parser.add_argument("--config", type=Path, default=Path("configs/hai_21_03.json"))
    parser.add_argument("--output", type=Path, default=Path("results/summaries/hai21_isolation_forest_target_reference.json"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--trees", type=int, default=500)
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--comparison-only", action="store_true", help="recompute paired statistics from an existing result file without rerunning the detector")
    parser.add_argument("--compact-existing-output", action="store_true", help="compact per-event output while retaining targets, top-10 ranks, and Hit@K")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    mapping_path = ROOT / "data/annotations/hai21_event_target_map.csv"
    mapping = pd.read_csv(mapping_path)
    mapping = mapping.loc[mapping["target_annotation_complete"].astype(bool)].copy()
    prior_path = ROOT / "results/summaries/hai21_identity_consistent_target_localization_seed42_46.json"
    if args.compact_existing_output:
        result = json.loads(output_path.read_text(encoding="utf-8"))
        keep = {
            "attack_id", "dataset_file", "event_index", "detected", "target_tags",
            "ranked_tags", "hit_at_1", "hit_at_3", "hit_at_5", "seed",
        }
        for row in result["target_events"]:
            row["ranked_tags"] = row["ranked_tags"][:10]
            for key in list(row):
                if key not in keep:
                    del row[key]
        result["stored_rank_depth"] = 10
        result["stored_rank_note"] = "Compact event records retain attack ID, source target tags, detection status, top 10 ranked tags, seed, and Hit@1/3/5. The uncompressed records and ranks are regenerated by rerunning the five-seed detector script."
        output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(output_path), "stored_rank_depth": 10, "events": len(result["target_events"])}))
        return
    if args.comparison_only:
        result = json.loads(output_path.read_text(encoding="utf-8"))
        result["paired_comparison_vs_pca_temporal"] = comparison_from_events(
            result["target_events"], prior_path, args.bootstrap_draws
        )
        result["random_ranking_benchmark"] = random_ranking_benchmark(
            result["target_events"], args.bootstrap_draws
        )
        output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result["paired_comparison_vs_pca_temporal"], indent=2))
        return
    file_records = []
    event_records = []
    seed_summaries = []
    runtime_started = time.perf_counter()

    for seed in args.seeds:
        print(f"seed {seed}: loading normal and held-out HAI files", flush=True)
        dataset = load_hai(config_path, max_train_rows=int(json.loads(config_path.read_text(encoding="utf-8-sig"))["training_rows_max"]), seed=seed)
        fit_started = time.perf_counter()
        detector = HAIIsolationForest(
            n_estimators=args.trees,
            max_samples=args.max_samples,
            random_state=seed,
            n_jobs=1,
        ).fit(dataset.train_x, dataset.train_dx)
        fit_seconds = time.perf_counter() - fit_started
        detector.model_.set_params(n_jobs=1)
        print(f"seed {seed}: fit complete ({fit_seconds:.1f}s); scoring normal validation", flush=True)
        validation_scores = score_in_batches(detector, dataset.validation.x, dataset.validation.dx)
        fpr_target = float(json.loads(config_path.read_text(encoding="utf-8-sig"))["validation_false_positive_rate"])
        threshold = float(np.quantile(validation_scores, 1.0 - fpr_target))
        validation_fpr = float(np.mean(validation_scores > threshold))
        target_events_this_seed = []
        total_y, total_scores = [], []
        for segment in dataset.tests:
            started = time.perf_counter()
            scores = score_in_batches(detector, segment.x, segment.dx)
            scoring_seconds = time.perf_counter() - started
            total_y.append(segment.attack)
            total_scores.append(scores)
            prediction = scores > threshold
            metrics = binary_metrics(segment.attack, scores, threshold)
            evt = event_metrics(segment.attack, prediction, segment.timestamps)
            per_file = mapping.loc[mapping["dataset_file"] == segment.name].copy()
            targets = first_alarm_targets(detector, segment, scores, threshold, dataset.feature_names, per_file)
            for row in targets:
                row["seed"] = int(seed)
            event_records.extend(targets)
            target_events_this_seed.extend(targets)
            print(f"seed {seed}: scored {segment.name} in {scoring_seconds:.1f}s", flush=True)
            file_records.append({
                "seed": int(seed),
                "file": segment.name,
                "point_metrics": metrics,
                "event_metrics": evt,
                "score_seconds": scoring_seconds,
                "batch_amortized_score_us_per_row": scoring_seconds * 1e6 / max(1, len(scores)),
                "target_eligible_events": len(targets),
                "target_events_detected": sum(bool(x["detected"]) for x in targets),
            })
        pooled_y = np.concatenate(total_y)
        pooled_scores = np.concatenate(total_scores)
        latency_ix = np.linspace(0, len(dataset.validation.x) - 1, min(500, len(dataset.validation.x)), dtype=int)
        latency_ms = []
        for ix in latency_ix:
            started = time.perf_counter()
            detector.score(dataset.validation.x[ix:ix + 1], dataset.validation.dx[ix:ix + 1])
            latency_ms.append((time.perf_counter() - started) * 1000.0)
        seed_summaries.append({
            "seed": int(seed),
            "training_rows": int(len(dataset.train_x)),
            "feature_count": len(dataset.feature_names),
            "threshold": threshold,
            "validation_observed_fpr": validation_fpr,
            "pooled_test_point_metrics": binary_metrics(pooled_y, pooled_scores, threshold),
            "fit_seconds": fit_seconds,
            "single_row_score_latency_p50_ms": float(np.percentile(latency_ms, 50)),
            "single_row_score_latency_p95_ms": float(np.percentile(latency_ms, 95)),
            "tree_parameter_bytes": detector.model_bytes(),
            "target_eligible_events": len(target_events_this_seed),
            "target_detected": sum(bool(x["detected"]) for x in target_events_this_seed),
            "unconditional_hit_at_1": float(np.mean([x["hit_at_1"] for x in target_events_this_seed])),
            "unconditional_hit_at_3": float(np.mean([x["hit_at_3"] for x in target_events_this_seed])),
            "unconditional_hit_at_5": float(np.mean([x["hit_at_5"] for x in target_events_this_seed])),
        })

    # Compare per-event seed-averaged outcomes against the already frozen PCA-TEMPORAL audit.
    comparisons = comparison_from_events(event_records, prior_path, args.bootstrap_draws)
    result = {
        "study": "HAI 21.03 external attack-target audit of a normal-only Isolation Forest reference",
        "status": "exploratory direct detector-family comparator; not a reproduction of RAID",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "data_manifest": dataset.manifest,
        "runtime": {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "protocol": {
            "training": "train1/train2 normal rows only; chronological random subsample per seed, maximum rows per config; no attack labels in fit",
            "threshold": "train3 normal scores only, configured 1% quantile",
            "detector": "500-tree IsolationForest, max_samples=512, contamination=auto; robust-scaled state and first difference concatenated; single-process scoring in 50,000-row batches",
            "explanation": "identity-preserving per-tag deletion: replace current tag with training-normal median/mode and recompute its first difference against the observed previous row",
            "target_truth": "only 29 manual attack intervals with complete exact CSV tag lists; attacks not detected count as localization misses",
            "comparison": "paired event-cluster comparison against PCA-TEMPORAL grouped-ablation outcomes averaged over the existing five runs; one event is one cluster",
            "not_matched_to_raid": "does not reproduce RAID's regime discovery, 330-feature physics pipeline, adaptive thresholds, rule-path prose, LLM fleet, or RAG evaluation",
        },
        "seed_summaries": seed_summaries,
        "files": file_records,
        "target_events": event_records,
        "paired_comparison_vs_pca_temporal": comparisons,
        "random_ranking_benchmark": random_ranking_benchmark(event_records, args.bootstrap_draws),
        "wall_seconds": time.perf_counter() - runtime_started,
        "limitations": [
            "All seeds reuse the same 29 unique attack events; seeds are repeated fits, not independent events.",
            "The 29 events are from five files in one HAI testbed and do not establish plant-level generalization.",
            "Detector-family comparison does not reproduce RAID or compare its rule-to-text fidelity endpoint.",
            "Isolation Forest grouped-ablation rankings measure score sensitivity to deletion, not causal relevance.",
            "Host timing is not edge-device or plant-deployment performance.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "seeds": len(seed_summaries), "events_per_seed": len(mapping), "summaries": seed_summaries, "paired_comparison": comparisons}, indent=2))


if __name__ == "__main__":
    main()

