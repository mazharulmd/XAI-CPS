from scripts.run_data4cyber_baseline import compact_result


def test_compact_result_retains_metrics_and_target_rankings_only():
    full = {
        "study": "test",
        "heldout_family_results": [{
            "heldout_family": "S3",
            "heldout_scenario": "scenario",
            "model": "IsolationForest-TEMPORAL",
            "normal_fpr_target": 0.01,
            "point_metrics": {"average_precision": 0.5},
            "explanation_events": [
                {"target_evaluation_eligible": False, "phase_at_event_start": "recon"},
                {"target_evaluation_eligible": True, "attack_event_index": 1,
                 "target_groups": ["meter"], "ranked_groups": ["other", "meter"],
                 "native_group_mean_ranked_groups": [], "evidence_tier": "exact",
                 "target_status": "exact", "detected": True, "first_alarm_utc": "t1",
                 "seconds_from_event_start_to_first_alarm": 0.0,
                 "target_hit_at_1": False, "target_hit_at_3": True, "target_hit_at_5": True,
                 "native_target_hit_at_1": None, "native_target_hit_at_3": None,
                 "native_target_hit_at_5": None,
                 "random_reference_unconditional_hit_probabilities": {"1": 0.1,"3": 0.3,"5": 0.5},
                 "top_group_score_reduction": 2.0,
                 "top5_jaccard_under_measurement_noise": {"0.01": 0.9,"0.05": 0.8},
                 "native_top5_jaccard_under_measurement_noise": {"0.01": None,"0.05": None},
                 "explanation_latency_ms": 1.2, "group_ablation_scores": {"meter": 2.0}},
            ],
        }],
        "first_alarm_explanation_summary": [{"mean_hit": 0.5}],
        "paired_family_comparisons": [{"contrast": "A-B", "p": 0.25,
                                        "family_differences": [{"family": "S3", "difference": 1.0}]}],
    }
    compact = compact_result(full)
    assert compact["heldout_family_results"][0]["point_metrics"] == {"average_precision": 0.5}
    assert "training_scenarios" not in compact["heldout_family_results"][0]
    assert len(compact["target_event_rankings"]) == 1
    columns = compact["target_event_ranking_columns"]
    row = dict(zip(columns, compact["target_event_rankings"][0]))
    assert row["ablation_top5"] == "other|meter"
    assert row["hit5"] is True
    assert compact["paired_family_comparisons"][0]["p"] == 0.25
    assert "family_differences" not in compact["paired_family_comparisons"][0]
    compact_again = compact_result(compact)
    assert compact_again["target_event_rankings"] == compact["target_event_rankings"]

