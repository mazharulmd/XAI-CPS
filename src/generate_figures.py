"""
Generates publication-quality figures for the XAI-CPS repository from empirical CSV results.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def generate_all_figures():
    res_dir = "/working_dir/c_d159ca20f093cd38/xai_cps_repo/results"
    fig_dir = "/working_dir/c_d159ca20f093cd38/xai_cps_repo/figures"
    os.makedirs(fig_dir, exist_ok=True)

    df_xai = pd.read_csv(os.path.join(res_dir, "xai_benchmark_results.csv"))
    df_abl = pd.read_csv(os.path.join(res_dir, "ablation_results.csv"))
    df_det = pd.read_csv(os.path.join(res_dir, "detection_results.csv"))

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "figure.dpi": 200
    })

    # 1. Fidelity-Latency-Localization Pareto Frontier on Edge Hardware (ST-GDN across SWaT, WADI, HAI)
    st_agg = df_xai[df_xai["Detector"] == "ST-GDN"].groupby("Explainer", as_index=False).mean(numeric_only=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    colors = {
        "Raw-MSE": "#64748B",
        "VG": "#0284C7",
        "IG": "#2563EB",
        "LIME": "#D97706",
        "KernelSHAP": "#DC2626",
        "ShaTS": "#7C3AED",
        "PI-CAX (Ours)": "#059669"
    }
    for _, row in st_agg.iterrows():
        m = row["Explainer"]
        x = row["Latency_Edge_ms"]
        y = row["Hit@3"] * 100.0
        size = (row["Fidelity_Comp"] ** 2) * 520
        ax.scatter(x, y, s=size, color=colors.get(m, "#333333"), alpha=0.88, edgecolors="black", linewidth=1.2, label=m)
        offset_y = 2.5 if m != "VG" else -4.5
        ax.annotate(f"{m}\n(Comp={row['Fidelity_Comp']:.2f})", (x, y),
                    xytext=(6, offset_y), textcoords="offset points", fontsize=8.5, fontweight="bold" if "PI-CAX" in m else "normal")

    ax.axvline(10.0, color="#DC2626", linestyle="--", linewidth=1.3, label="10 ms Hard Real-Time Edge Deadline")
    ax.set_xscale("log")
    ax.set_xlim(0.4, 220.0)
    ax.set_ylim(18.0, 102.0)
    ax.set_xlabel("Edge Execution Latency on ARM Cortex-A72 Profile (ms, log scale)")
    ax.set_ylabel("Root-Cause Localization Top-3 Accuracy Hit@3 (%)")
    ax.set_title("Fidelity-Localization-Latency Pareto Frontier Across Multi-Stage CPS Benchmarks")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig1_pareto_frontier.png"), bbox_inches="tight")
    plt.close(fig)

    # 2. Sensor vs. Actuator vs. Multi-Point Attack Attribution Breakdown (AvgRank lower is better)
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    order = ["KernelSHAP", "LIME", "ShaTS", "Raw-MSE", "VG", "IG", "PI-CAX (Ours)"]
    st_ord = st_agg.set_index("Explainer").loc[order].reset_index()
    x_pos = np.arange(len(order))
    width = 0.26
    ax.bar(x_pos - width, st_ord["Sensor_AvgRank"], width, label="Continuous Sensor Attacks", color="#3B82F6")
    ax.bar(x_pos, st_ord["Actuator_AvgRank"], width, label="Discrete Actuator Hijacks", color="#EF4444")
    ax.bar(x_pos + width, st_ord["Multi_AvgRank"], width, label="Multi-Point Stealth Attacks", color="#8B5CF6")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(order, rotation=18, ha="right")
    ax.set_ylabel("Normalized Average Rank (AvgRank, Lower is Better)")
    ax.set_title("Root-Cause Attribution Error Across Sensor, Actuator, and Multi-Point CPS Attacks")
    ax.legend(frameon=True)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig2_actuator_vs_sensor_avgrank.png"), bbox_inches="tight")
    plt.close(fig)

    # 3. Ablation Study Bar Chart
    abl_agg = df_abl.groupby("Configuration", as_index=False, sort=False).mean(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    y_pos = np.arange(len(abl_agg))
    ax.barh(y_pos - 0.18, abl_agg["Hit@3"] * 100.0, height=0.34, label="Hit@3 Localization (%)", color="#059669")
    ax.barh(y_pos + 0.18, abl_agg["Adv_Rob"] * 100.0, height=0.34, label="Adversarial Spoofing Robustness (%)", color="#0284C7")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(abl_agg["Configuration"])
    ax.invert_yaxis()
    ax.set_xlabel("Percentage Score (%)")
    ax.set_title("Ablation Study of PI-CAX Architectural Components (Mean Across SWaT, WADI, HAI)")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig3_ablation_study.png"), bbox_inches="tight")
    plt.close(fig)
    print("Generated all figures in", fig_dir)

if __name__ == "__main__":
    generate_all_figures()
