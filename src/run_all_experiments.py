"""
Master Experimental Runner for XAI-IDS-CPS Q1 Research Study
============================================================
Executes all experiments across 3 CPS Benchmarks (SWaT, WADI, HAI),
4 Intrusion Detection Architectures (VAR-SSM, TCN-AE, GRU-Seq, ST-GDN + PI-CAX Gated),
7 XAI Methods (Raw-MSE, VG, IG, LIME, KernelSHAP, ShaTS, PI-CAX),
7 Ablation Configurations, and Non-Parametric Statistical Significance Tests.
Saves all JSON/CSV tables to xai_cps_repo/results/.
"""
import os
import json
import numpy as np
import pandas as pd
from scipy import stats

from xai_cps_repo.src.cps_benchmarks import build_cps_benchmark
from xai_cps_repo.src.detectors import VARSSMDetector, TCNAEDetector, GRUSeqDetector, STGDNDetector
from xai_cps_repo.src.explainers import CPSExplainerSuite
from xai_cps_repo.src.evaluation import (
    evaluate_detection, evaluate_explainer_on_attacks, compute_cliffs_delta
)

def main():
    out_dir = "/working_dir/c_d159ca20f093cd38/xai_cps_repo/results"
    os.makedirs(out_dir, exist_ok=True)

    datasets = ["SWaT", "WADI", "HAI"]
    detector_classes = [
        ("VAR-SSM", VARSSMDetector),
        ("TCN-AE", TCNAEDetector),
        ("GRU-Seq", GRUSeqDetector),
        ("ST-GDN", STGDNDetector),
    ]

    det_rows = []
    xai_rows = []
    all_attack_records = []
    ablation_rows = []

    for ds_name in datasets:
        bench = build_cps_benchmark(ds_name, seed=42)
        print(f"\n==================== DATASET: {ds_name} ====================")

        trained_dets = {}
        for det_name, det_cls in detector_classes:
            det = det_cls(window_size=10).fit(bench.X_train, bench.X_val, bench.feature_types, bench.stage_map)
            trained_dets[det_name] = det
            res = evaluate_detection(det, bench)
            res["Dataset"] = ds_name
            res["Detector"] = det_name
            det_rows.append(res)
            print(f"  [Det] {det_name:12s} | F1_PA={res['F1_PA']:.4f} | F1_Raw={res['F1_Raw']:.4f} | AUROC={res['AUROC']:.4f} | FPR={res['FPR']:.4f} | Delay={res['Mean_Delay_s']:.2f}s")

        # Evaluate PI-CAX Physics-Informed Coherence Gating on top of ST-GDN
        st_det = trained_dets["ST-GDN"]
        st_suite = CPSExplainerSuite(st_det, bench.X_train, bench.feature_types, bench.stage_map,
                                     bench.safety_bounds, bench.max_slew_rate, seed=42)
        res_picax_det = evaluate_detection(st_det, bench, suite=st_suite, use_picax_gating=True)
        res_picax_det["Dataset"] = ds_name
        res_picax_det["Detector"] = "ST-GDN + PI-CAX (Ours)"
        det_rows.append(res_picax_det)
        print(f"  [Det] {'ST-GDN+PI-CAX':12s} | F1_PA={res_picax_det['F1_PA']:.4f} | F1_Raw={res_picax_det['F1_Raw']:.4f} | AUROC={res_picax_det['AUROC']:.4f} | FPR={res_picax_det['FPR']:.4f} | Delay={res_picax_det['Mean_Delay_s']:.2f}s")

        # Evaluate all 7 XAI methods across all 4 detectors on SWaT, and on ST-GDN + GRU-Seq across WADI & HAI
        target_det_names = ["VAR-SSM", "TCN-AE", "GRU-Seq", "ST-GDN"] if ds_name == "SWaT" else ["GRU-Seq", "ST-GDN"]
        for det_name in target_det_names:
            det = trained_dets[det_name]
            suite = CPSExplainerSuite(det, bench.X_train, bench.feature_types, bench.stage_map,
                                      bench.safety_bounds, bench.max_slew_rate, seed=42)
            methods = [
                ("Raw-MSE", suite.explain_raw_mse),
                ("VG", suite.explain_vg),
                ("IG", suite.explain_ig),
                ("LIME", suite.explain_lime),
                ("KernelSHAP", suite.explain_kernel_shap),
                ("ShaTS", suite.explain_shats),
                ("PI-CAX (Ours)", suite.explain_pi_cax),
            ]
            for m_name, m_fn in methods:
                summary, recs = evaluate_explainer_on_attacks(suite, bench, m_name, m_fn)
                summary["Dataset"] = ds_name
                summary["Detector"] = det_name
                summary["Explainer"] = m_name
                xai_rows.append(summary)
                for r in recs:
                    r["detector"] = det_name
                    all_attack_records.append(r)
                if det_name == "ST-GDN":
                    print(f"  [XAI:{det_name}] {m_name:14s} | Hit@1={summary['Hit@1']:.3f} | Hit@3={summary['Hit@3']:.3f} | AvgRank={summary['AvgRank']:.3f} | ActRank={summary['Actuator_AvgRank']:.3f} | Comp={summary['Fidelity_Comp']:.3f} | Stab={summary['Stability']:.3f} | Edge={summary['Latency_Edge_ms']:.2f}ms")

        # Run 7-Configuration Ablation Study on ST-GDN across all 3 datasets
        ablation_configs = [
            ("Full PI-CAX (Ours)", lambda w: st_suite.explain_pi_cax(w, use_causal_decoupling=True, use_manifold_shapley=True, use_phys_envelope=True, use_tier2_refinement=True)),
            ("w/o Causal Decoupling (PC-SCM)", lambda w: st_suite.explain_pi_cax(w, use_causal_decoupling=False, use_manifold_shapley=True, use_phys_envelope=True, use_tier2_refinement=True)),
            ("w/o Manifold Precision Shapley", lambda w: st_suite.explain_pi_cax(w, use_causal_decoupling=True, use_manifold_shapley=False, use_phys_envelope=True, use_tier2_refinement=True)),
            ("w/o Physical Safety/Slew Envelope", lambda w: st_suite.explain_pi_cax(w, use_causal_decoupling=True, use_manifold_shapley=True, use_phys_envelope=False, use_tier2_refinement=True)),
            ("w/o Tier-2 Counterfactual Verification (Tier-1 Only)", lambda w: st_suite.explain_pi_cax(w, use_causal_decoupling=True, use_manifold_shapley=True, use_phys_envelope=True, use_tier2_refinement=False)),
            ("PC-SCM + Tier-1 Only (No Envelope/Tier-2)", lambda w: st_suite.explain_pi_cax(w, use_causal_decoupling=True, use_manifold_shapley=False, use_phys_envelope=False, use_tier2_refinement=False)),
            ("Base Detector Residual (Raw-MSE)", st_suite.explain_raw_mse),
        ]
        for ab_name, ab_fn in ablation_configs:
            ab_sum, _ = evaluate_explainer_on_attacks(st_suite, bench, ab_name, ab_fn)
            ab_sum["Dataset"] = ds_name
            ab_sum["Configuration"] = ab_name
            ablation_rows.append(ab_sum)

    df_det = pd.DataFrame(det_rows)
    df_xai = pd.DataFrame(xai_rows)
    df_recs = pd.DataFrame(all_attack_records)
    df_abl = pd.DataFrame(ablation_rows)

    df_det.to_csv(os.path.join(out_dir, "detection_results.csv"), index=False)
    df_xai.to_csv(os.path.join(out_dir, "xai_benchmark_results.csv"), index=False)
    df_recs.to_csv(os.path.join(out_dir, "per_attack_xai_records.csv"), index=False)
    df_abl.to_csv(os.path.join(out_dir, "ablation_results.csv"), index=False)

    # Statistical Significance Tests (PI-CAX vs each baseline across all attack instances on ST-GDN)
    st_recs = df_recs[df_recs["detector"] == "ST-GDN"]
    picax_sub = st_recs[st_recs["method"] == "PI-CAX (Ours)"].sort_values(["dataset", "attack_id"])
    stat_rows = []
    for base_m in ["Raw-MSE", "VG", "IG", "LIME", "KernelSHAP", "ShaTS"]:
        base_sub = st_recs[st_recs["method"] == base_m].sort_values(["dataset", "attack_id"])
        # Paired Wilcoxon signed-rank test on AvgRank (lower is better), Fidelity_Comp (higher is better), Stability (higher is better), and Latency
        w_rank, p_rank = stats.wilcoxon(picax_sub["avg_rank"].values, base_sub["avg_rank"].values, alternative="less", zero_method="zsplit")
        w_comp, p_comp = stats.wilcoxon(picax_sub["comp_score"].values, base_sub["comp_score"].values, alternative="greater", zero_method="zsplit")
        w_stab, p_stab = stats.wilcoxon(picax_sub["stability"].values, base_sub["stability"].values, alternative="greater", zero_method="zsplit")
        delta_rank = compute_cliffs_delta(base_sub["avg_rank"].values, picax_sub["avg_rank"].values)
        delta_comp = compute_cliffs_delta(picax_sub["comp_score"].values, base_sub["comp_score"].values)
        speedup = float(base_sub["edge_latency_ms"].mean() / (picax_sub["edge_latency_ms"].mean() + 1e-9))
        stat_rows.append({
            "Comparison": f"PI-CAX vs. {base_m}",
            "AvgRank_Diff": round(float(picax_sub["avg_rank"].mean() - base_sub["avg_rank"].mean()), 4),
            "Wilcoxon_p_AvgRank": float(f"{p_rank:.3e}"),
            "Cliffs_Delta_Rank": round(delta_rank, 3),
            "Comp_Diff": round(float(picax_sub["comp_score"].mean() - base_sub["comp_score"].mean()), 4),
            "Wilcoxon_p_Comp": float(f"{p_comp:.3e}"),
            "Cliffs_Delta_Comp": round(delta_comp, 3),
            "Wilcoxon_p_Stab": float(f"{p_stab:.3e}"),
            "Edge_Speedup_x": round(speedup, 2)
        })
    df_stats = pd.DataFrame(stat_rows)
    df_stats.to_csv(os.path.join(out_dir, "statistical_tests.csv"), index=False)

    print("\n==================== AGGREGATE XAI SUMMARY (ST-GDN ACROSS ALL DATASETS) ====================")
    st_xai = df_xai[df_xai["Detector"] == "ST-GDN"].groupby("Explainer", as_index=False).mean(numeric_only=True)
    print(st_xai[["Explainer", "Hit@1", "Hit@3", "AvgRank", "Actuator_AvgRank", "Stage_Hit@1", "Fidelity_Comp", "Stability", "Adv_Rob", "Latency_Edge_ms", "DCR_10ms"]].to_string(index=False))

    print("\n==================== ABLATION SUMMARY (MEAN ACROSS ALL DATASETS) ====================")
    abl_agg = df_abl.groupby("Configuration", as_index=False, sort=False).mean(numeric_only=True)
    print(abl_agg[["Configuration", "Hit@1", "Hit@3", "AvgRank", "Actuator_AvgRank", "Fidelity_Comp", "Stability", "Adv_Rob", "Latency_Edge_ms"]].to_string(index=False))

    print("\n==================== STATISTICAL SIGNIFICANCE TESTS ====================")
    print(df_stats.to_string(index=False))

if __name__ == "__main__":
    main()
